import logging
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchaudio

from mmaudio.eval_utils import (ModelConfig, all_model_cfg, generate, load_video, make_video,
                                setup_eval_logging)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils

# 启用 TensorFloat-32 (TF32) 加速计算（如果硬件支持）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 初始化日志记录器
log = logging.getLogger()


@torch.inference_mode()
def main():
    # 设置评估日志记录
    setup_eval_logging()

    # 定义命令行参数解析器
    parser = ArgumentParser()
    parser.add_argument('--variant',
                        type=str,
                        default='large_44k_v2',
                        help='small_16k, small_44k, medium_44k, large_44k, large_44k_v2')
    parser.add_argument('--video', type=Path, help='Path to the video file')
    parser.add_argument('--prompt', type=str, help='Input prompt', default='')
    parser.add_argument('--negative_prompt', type=str, help='Negative prompt', default='')
    parser.add_argument('--duration', type=float, default=8.0)
    parser.add_argument('--cfg_strength', type=float, default=4.5)
    parser.add_argument('--num_steps', type=int, default=25)

    parser.add_argument('--mask_away_clip', action='store_true')

    parser.add_argument('--output_path', type=Path, help='Output directory', default='./output')
    parser.add_argument('--output_name', type=str, help='Output file name', default='')
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--skip_video_composite', action='store_true')
    parser.add_argument('--full_precision', action='store_true')

    args = parser.parse_args()

    # 验证选择的模型变体是否有效
    if args.variant not in all_model_cfg:
        raise ValueError(f'Unknown model variant: {args.variant}')
    model: ModelConfig = all_model_cfg[args.variant]
    # 确保模型权重已下载
    model.download_if_needed()
    # 模型的序列配置
    seq_cfg = model.seq_cfg

    # 如果提供了视频路径，则展开并验证路径
    if args.video:
        video_path: Path = Path(args.video).expanduser()
    else:
        video_path = None

    # 提取其他参数
    prompt: str = args.prompt
    negative_prompt: str = args.negative_prompt
    output_dir: str = args.output_path.expanduser()
    output_name: str = args.output_name
    seed: int = args.seed
    num_steps: int = args.num_steps
    duration: float = args.duration
    cfg_strength: float = args.cfg_strength
    skip_video_composite: bool = args.skip_video_composite
    mask_away_clip: bool = args.mask_away_clip

    # 确定使用的设备（CUDA、MPS 或 CPU）
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        log.warning('CUDA/MPS are not available, running on CPU')
    dtype = torch.float32 if args.full_precision else torch.bfloat16

    # 如果输出目录不存在，则创建
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载预训练模型
    net: MMAudio = get_my_mmaudio(model.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))
    log.info(f'Loaded weights from {model.model_path}')

    # 设置随机数生成器以确保结果可复现
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    # 初始化流匹配模块以进行推理
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

    # 初始化音频生成的特征工具
    feature_utils = FeaturesUtils(tod_vae_ckpt=model.vae_path,
                                  synchformer_ckpt=model.synchformer_ckpt,
                                  enable_conditions=True,
                                  mode=model.mode,
                                  bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
                                  need_vae_encoder=False)
    feature_utils = feature_utils.to(device, dtype).eval()

    # 如果提供了视频，则进行处理
    if video_path is not None:
        log.info(f'Using video {video_path}')
        video_info = load_video(video_path, duration)
        clip_frames = video_info.clip_frames
        sync_frames = video_info.sync_frames
        duration = video_info.duration_sec
        if mask_away_clip:
            clip_frames = None
        else:
            clip_frames = clip_frames.unsqueeze(0)
        sync_frames = sync_frames.unsqueeze(0)
    else:
        log.info('No video provided -- text-to-audio mode')
        clip_frames = sync_frames = None

    # 使用生成的时长更新序列配置
    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    # 记录提示信息以便调试
    log.info(f'Prompt: {prompt}')
    log.info(f'Negative prompt: {negative_prompt}')

    # 使用模型和工具生成音频
    audios = generate(clip_frames,
                      sync_frames, [prompt],
                      negative_text=[negative_prompt],
                      feature_utils=feature_utils,
                      net=net,
                      fm=fm,
                      rng=rng,
                      cfg_strength=cfg_strength)
    audio = audios.float().cpu()[0]

    # 将生成的音频保存到输出目录
    if video_path is not None:
        if output_name:
            save_path = output_dir / f'{output_name}.flac'
        else:
            save_path = output_dir / f'{video_path.stem}.flac'
    else:
        if output_name:
            save_path = output_dir / f'{output_name}.flac'
        else:
            safe_filename = prompt.replace(' ', '_').replace('/', '_').replace('.', '')
            save_path = output_dir / f'{safe_filename}.flac'
    torchaudio.save(save_path, audio, seq_cfg.sampling_rate)
    log.info(f'Audio saved to:{save_path}')

    # 如果提供了视频且未跳过合成步骤，则保存带音频的视频
    if video_path is not None and not skip_video_composite:
        if output_name:
            video_save_path = output_dir / f'{output_name}.mp4'
        else:
            video_save_path = output_dir / f'{video_path.stem}.mp4'
        make_video(video_info, video_save_path, audio, sampling_rate=seq_cfg.sampling_rate)
        log.info(f'Video saved to {output_dir / video_save_path}')

    log.info('Memory usage: %.2f GB', torch.cuda.max_memory_allocated() / (2**30))


if __name__ == '__main__':
    main()
