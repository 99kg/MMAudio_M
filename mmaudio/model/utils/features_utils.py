from typing import Literal, Optional

import open_clip
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from open_clip import create_model_from_pretrained
from torchvision.transforms import Normalize

from mmaudio.ext.autoencoder import AutoEncoderModule
from mmaudio.ext.mel_converter import get_mel_converter
from mmaudio.ext.synchformer import Synchformer
from mmaudio.model.utils.distributions import DiagonalGaussianDistribution


# 定义一个函数，用于修改 CLIP 模型的 encode_text 方法，使其输出最后的隐藏状态
def patch_clip(clip_model):
    # 一个 hack 方法，用于让 CLIP 模型输出最后的隐藏状态
    # 参考链接：https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/model.py#L269
    def new_encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        # 获取 token 嵌入
        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        # 添加位置嵌入并通过 transformer
        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        return F.normalize(x, dim=-1) if normalize else x

    # 替换 CLIP 模型的 encode_text 方法
    clip_model.encode_text = new_encode_text.__get__(clip_model)
    return clip_model


# 定义一个工具类，用于处理特征提取和编码
class FeaturesUtils(nn.Module):

    def __init__(
        self,
        *,
        tod_vae_ckpt: Optional[str] = None,
        bigvgan_vocoder_ckpt: Optional[str] = None,
        synchformer_ckpt: Optional[str] = None,
        enable_conditions: bool = True,
        mode=Literal['16k', '44k'],
        need_vae_encoder: bool = True,
    ):
        super().__init__()

        if enable_conditions:
            # self.clip_model = create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14-384',
            #                                                return_transform=False)
            # 当前脚本所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 构建相对路径
            clip_model_path = os.path.join(current_dir, '../../../local_files/apple/DFN5B-CLIP-ViT-H-14-384')
            # 替换路径中的反斜杠为正斜杠
            clip_model_path = clip_model_path.replace('\\', '/')
            # 使用绝对路径
            abs_clip_model_path = os.path.abspath(clip_model_path)
            self.clip_model = create_model_from_pretrained(f'local-dir:{abs_clip_model_path}', return_transform=False)

            # 设置 CLIP 模型的预处理参数
            self.clip_preprocess = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                             std=[0.26862954, 0.26130258, 0.27577711])
            self.clip_model = patch_clip(self.clip_model)

            # 加载 Synchformer 模型
            self.synchformer = Synchformer()
            self.synchformer.load_state_dict(
                torch.load(synchformer_ckpt, weights_only=True, map_location='cpu'))

            # 初始化 tokenizer
            # 与 'ViT-H-14' 相同
            self.tokenizer = open_clip.get_tokenizer('ViT-H-14-378-quickgelu')
        else:
            self.clip_model = None
            self.synchformer = None
            self.tokenizer = None

        if tod_vae_ckpt is not None:
            # 初始化 Mel 转换器和 AutoEncoder 模块
            self.mel_converter = get_mel_converter(mode)
            self.tod = AutoEncoderModule(vae_ckpt_path=tod_vae_ckpt,
                                         vocoder_ckpt_path=bigvgan_vocoder_ckpt,
                                         mode=mode,
                                         need_vae_encoder=need_vae_encoder)
        else:
            self.tod = None

    # 编译模型以提高推理速度
    def compile(self):
        if self.clip_model is not None:
            self.clip_model.encode_image = torch.compile(self.clip_model.encode_image)
            self.clip_model.encode_text = torch.compile(self.clip_model.encode_text)
        if self.synchformer is not None:
            self.synchformer = torch.compile(self.synchformer)
        self.decode = torch.compile(self.decode)
        self.vocode = torch.compile(self.vocode)

    # 设置模型为评估模式
    def train(self, mode: bool) -> None:
        return super().train(False)

    @torch.inference_mode()
    def encode_video_with_clip(self, x: torch.Tensor, batch_size: int = -1) -> torch.Tensor:
        assert self.clip_model is not None, 'CLIP is not loaded'
        # x: (B, T, C, H, W) H/W: 384
        b, t, c, h, w = x.shape
        assert c == 3 and h == 384 and w == 384
        x = self.clip_preprocess(x)
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        outputs = []
        if batch_size < 0:
            batch_size = b * t
        for i in range(0, b * t, batch_size):
            outputs.append(self.clip_model.encode_image(x[i:i + batch_size], normalize=True))
        x = torch.cat(outputs, dim=0)
        # x = self.clip_model.encode_image(x, normalize=True)
        x = rearrange(x, '(b t) d -> b t d', b=b)
        return x

    @torch.inference_mode()
    def encode_video_with_sync(self, x: torch.Tensor, batch_size: int = -1) -> torch.Tensor:
        assert self.synchformer is not None, 'Synchformer is not loaded'
        # x: (B, T, C, H, W) H/W: 384

        b, t, c, h, w = x.shape
        assert c == 3 and h == 224 and w == 224

        # 将视频分段
        segment_size = 16
        step_size = 8
        num_segments = (t - segment_size) // step_size + 1
        segments = []
        for i in range(num_segments):
            segments.append(x[:, i * step_size:i * step_size + segment_size])
        x = torch.stack(segments, dim=1)  # (B, S, T, C, H, W)

        outputs = []
        if batch_size < 0:
            batch_size = b
        x = rearrange(x, 'b s t c h w -> (b s) 1 t c h w')
        for i in range(0, b * num_segments, batch_size):
            outputs.append(self.synchformer(x[i:i + batch_size]))
        x = torch.cat(outputs, dim=0)
        x = rearrange(x, '(b s) 1 t d -> b (s t) d', b=b)
        return x

    @torch.inference_mode()
    def encode_text(self, text: list[str]) -> torch.Tensor:
        assert self.clip_model is not None, 'CLIP is not loaded'
        assert self.tokenizer is not None, 'Tokenizer is not loaded'
        # 将文本编码为 token
        tokens = self.tokenizer(text).to(self.device)
        return self.clip_model.encode_text(tokens, normalize=True)

    @torch.inference_mode()
    def encode_audio(self, x) -> DiagonalGaussianDistribution:
        assert self.tod is not None, 'VAE is not loaded'
        # 将音频转换为 Mel 频谱
        mel = self.mel_converter(x)
        dist = self.tod.encode(mel)

        return dist

    @torch.inference_mode()
    def vocode(self, mel: torch.Tensor) -> torch.Tensor:
        assert self.tod is not None, 'VAE is not loaded'
        return self.tod.vocode(mel)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        assert self.tod is not None, 'VAE is not loaded'
        return self.tod.decode(z.transpose(1, 2))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
