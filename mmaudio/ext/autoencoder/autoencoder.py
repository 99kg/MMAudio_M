from typing import Literal, Optional

import os
import torch
import torch.nn as nn

from mmaudio.ext.autoencoder.vae import VAE, get_my_vae
from mmaudio.ext.bigvgan import BigVGAN
from mmaudio.ext.bigvgan_v2.bigvgan import BigVGAN as BigVGANv2
from mmaudio.model.utils.distributions import DiagonalGaussianDistribution


class AutoEncoderModule(nn.Module):

    def __init__(self,
                 *,
                 vae_ckpt_path,
                 vocoder_ckpt_path: Optional[str] = None,
                 mode: Literal['16k', '44k'],
                 need_vae_encoder: bool = True):
        super().__init__()
        self.vae: VAE = get_my_vae(mode).eval()
        vae_state_dict = torch.load(vae_ckpt_path, weights_only=True, map_location='cpu')
        self.vae.load_state_dict(vae_state_dict)
        self.vae.remove_weight_norm()

        if mode == '16k':
            assert vocoder_ckpt_path is not None
            self.vocoder = BigVGAN(vocoder_ckpt_path).eval()
        elif mode == '44k':
            #self.vocoder = BigVGANv2.from_pretrained('nvidia/bigvgan_v2_44khz_128band_512x',
            #                                         use_cuda_kernel=False)
            # 当前脚本所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 构建相对路径
            vocoder_model_path = os.path.join(current_dir, '../../../local_files/nvida/bigvgan_v2_44khz_128band_512x')
            # 替换路径中的反斜杠为正斜杠
            vocoder_model_path = vocoder_model_path.replace('\\', '/')
            # 使用绝对路径
            abs_vocoder_model_path = os.path.abspath(vocoder_model_path)
            # 替换路径中的反斜杠为正斜杠
            abs_vocoder_model_path = abs_vocoder_model_path.replace('\\', '/')

            self.vocoder = BigVGANv2.from_pretrained(f'{abs_vocoder_model_path}', use_cuda_kernel=False, local_files_only=True)
            self.vocoder.remove_weight_norm()
        else:
            raise ValueError(f'Unknown mode: {mode}')

        for param in self.parameters():
            param.requires_grad = False

        if not need_vae_encoder:
            del self.vae.encoder

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        return self.vae.encode(x)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z)

    @torch.inference_mode()
    def vocode(self, spec: torch.Tensor) -> torch.Tensor:
        return self.vocoder(spec)
