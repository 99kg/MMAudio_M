# MMAudio_M — 环境搭建说明（中文版）

本文档基于项目在 Windows PowerShell 下的安装日志整理，包含从创建虚拟环境到安装依赖、常见问题与验证步骤。所有命令均针对 PowerShell（Windows）环境，示例来自实际安装记录。

## 一、前提条件

- 操作系统：Windows（本文档基于 PowerShell 的日志）。
- Python：日志中使用的 wheel 显示为 `cp313`，说明使用的是 Python 3.13（若你使用其他 Python 版本，注意选择与之匹配的二进制包）。
- GPU & 驱动：日志中通过 `nvidia-smi` 检测到 NVIDIA GPU（示例：RTX 4060），Driver Version: 566.24，CUDA Version: 12.7；实际安装的 PyTorch 二进制为 cu118（CUDA 11.8）构建，请确保驱动兼容所需 CUDA 版本。
- 建议已安装 Microsoft Visual C++ Build Tools（用于编译某些依赖）。

## 二、常见权限问题（PowerShell）

日志开头出现：

> 无法加载文件 ... profile.ps1，因为在此系统上禁止运行脚本。

解决方法（以当前用户允许远程签名脚本）：

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# 系统会提示是否更改，选择 Y
```

说明：如果你不想改动全局策略，也可以只在当前会话运行脚本：

```powershell
powershell -ExecutionPolicy Bypass -File .\some-script.ps1
```

## 三、标准安装步骤（按日志整理）

1. 进入项目目录：

```powershell
cd C:\Users\18071\Desktop\MMAudio_M
```

2. 创建并激活虚拟环境：

```powershell
python -m venv mmaudio_env
.\mmaudio_env\Scripts\Activate.ps1
# 激活后命令行会显示 (mmaudio_env)
```

3. 验证 GPU（如机器有 NVIDIA）：

```powershell
nvidia-smi
```

4. 安装 PyTorch（使用日志中记录的特定版本）

注意：根据你的要求与日志记录，本项目在该环境中使用了日志中的指定版本 —— 请使用下面的版本以保证一致性。

日志中使用（强制使用这些版本）：

```powershell
# 使用日志中相同的 wheel 与版本（cu118 构建）。
pip install "torch==2.7.1+cu118" "torchvision==0.22.1+cu118" "torchaudio==2.7.1+cu118" --index-url https://download.pytorch.org/whl/cu118 --upgrade
```

说明：如果你的 CUDA 版本不同（或使用 CPU），请参考 PyTorch 官方安装页选择合适的 index-url 或 wheel；但如果要严格复现日志环境，请使用上面指定的版本。

5. 升级 pip（推荐使用 python -m pip）：

日志中演示并建议使用：

```powershell
python.exe -m pip install --upgrade pip
```

如果命令被提示需要以特定方式修改 pip，请按提示使用 python -m pip。

6. 在项目目录安装（可编辑安装）：

```powershell
python.exe -m pip install -e .
```

该命令会安装项目及其依赖（日志中显示大量依赖被下载并成功安装，例如 `librosa`, `gradio`, `huggingface-hub` 等）。日志中也使用了国内镜像（如清华镜像）来加速包下载：`https://pypi.tuna.tsinghua.edu.cn/simple`。如果网络速度慢，可考虑临时加参数 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。

7. 指定 transformers 与 huggingface-hub 版本（使用日志中的版本）

日志中最终使用的版本为 `transformers-4.57.1` 和 `huggingface-hub-0.36.0`。若要与该环境一致，请按下列命令安装指定版本：

```powershell
pip install "transformers==4.57.1" "huggingface-hub==0.36.0"
```

注意：pip 包名为 `huggingface-hub`（中间有连字符）。如果你在安装过程中遇到依赖冲突，请先卸载冲突版本再安装指定版本。

8. 期望的权重与扩展权重目录结构（示例）

下面给出两个示例：完整（包含多个已知模型权重）和最小（只包含推荐模型的最小集合）。请把模型权重放到相应目录下以便项目加载。
https://huggingface.co/hkchengrex/MMAudio/tree/main

预期目录结构（完整）:

```
MMAudio
├── ext_weights
│   ├── best_netG.pt
│   ├── synchformer_state_dict.pth
│   ├── v1-16.pth
│   └── v1-44.pth
├── weights
│   ├── mmaudio_small_16k.pth
│   ├── mmaudio_small_44k.pth
│   ├── mmaudio_medium_44k.pth
│   ├── mmaudio_large_44k.pth
│   └── mmaudio_large_44k_v2.pth
└── ...
```

预期目录结构（最小，推荐模型）:

```
MMAudio
├── ext_weights
│   ├── synchformer_state_dict.pth
│   └── v1-44.pth
├── weights
│   └── mmaudio_large_44k_v2.pth
└── ...
```

9. 预训练模型配置文件本地化配置

为了更好地管理预训练模型的配置文件，建议按如下结构组织本地配置文件：

预期目录结构（完整）：

```
MMAudio
├── local_file
│   ├── apple
│         ├── DFN5B-CLIP-ViT-H-14-384
│              ├── open_clip_config.json
│              ├── open_clip_pytorch_model.bin
│   ├── nvida
│         ├── bigvgan_v2_44khz_128band_512x
│              ├── bigvgan_generator.pt
│              ├── config.json
└── ...
```

说明：此目录结构用于存放各个预训练模型的配置文件，按照供应商（如 apple、nvidia）分类组织，便于管理和维护。

## 四、验证安装

在虚拟环境中运行以下命令来验证关键组件：

```powershell
# 检查 Python & pip
python --version
pip --version

# 检查 PyTorch 版本与 CUDA 可用性
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available())"

# 检查项目能否被导入（示例）
python -c "import mmaudio; print('mmaudio imported')"
```

如果上述命令没有报错，说明核心安装成功。

## 五、常见问题与解决办法

1. "在此系统上禁止运行脚本"（ExecutionPolicy）
   - 见上文，使用 `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`。

2. pip 无法修改或提示请用 python -m pip：
   - 使用 `python -m pip install --upgrade pip` 来保证 pip 在当前 Python 解释器下被正确更新。

3. GPU/CUDA 不可用或 PyTorch 与 CUDA 版本不匹配：
   - 使用 `nvidia-smi` 查看驱动与 CUDA 版本。按需选择 PyTorch 的对应 CUDA 构建（例如 cu118）。若需 CPU-only 版本，可安装 `pip install torch --index-url https://download.pytorch.org/whl/cpu`（或参考官方安装页）。

4. 某些依赖编译失败（例如需要 C 编译器）：
   - 请安装 Visual C++ Build Tools（或适配的编译工具），Windows 上建议安装 "Microsoft Build Tools"。

5. 网络问题导致下载失败：
   - 使用国内镜像：在 pip 命令中加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`，或者配置 pip 的 `pip.ini` 来长期使用镜像。

## 六、日志中注意到的版本与信息（供参考）

- PyTorch: torch-2.7.1+cu118 (日志安装)
- torchvision: 0.22.1+cu118
- torchaudio: 2.7.1+cu118
- Python wheel tags in日志：cp313 -> Python 3.13
- NVIDIA 驱动: 566.24，CUDA Version: 12.7（nvidia-smi 输出）
- 许多依赖（librosa、gradio、transformers 等）被成功安装。

## 七、附录：常用命令速查（PowerShell）

```powershell
# 进入项目目录
cd C:\Users\18071\Desktop\MMAudio_M

# 允许当前用户运行已签名脚本
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 创建并激活虚拟环境
python -m venv mmaudio_env
.\mmaudio_env\Scripts\Activate.ps1

# 检查 GPU
nvidia-smi

# 安装 PyTorch（示例，cu118）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade

# 升级 pip（推荐使用 python -m）
python -m pip install --upgrade pip

# 安装项目（可编辑模式）
python -m pip install -e .

# 检查 PyTorch
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 八、后续建议

- 若你使用的是 Python 3.11 或其它版本，请根据 Python 版本选择合适的 wheel（日志中为 cp313，即 Python 3.13）。
- 为减少网络问题，可以提前下载大包（如 PyTorch 的 wheel）并离线安装。
- 若需部署到没有 GPU 的机器，安装 CPU-only 的 PyTorch 构建。

---

如果你希望我把 README 中的某些部分改成更简洁的安装脚本（例如一个 PowerShell 脚本自动执行可选步骤），或把说明改写为针对其他 Python/CUDA 版本的变体，我可以继续生成并把脚本加入仓库。

## 九、安装日志（结尾摘录）

下面为安装日志结尾部分的原始输出摘录，用于确认哪些包被成功安装及最终目录操作：

```log
Successfully installed absl-py-2.3.1 aiofiles-24.1.0 annotated-doc-0.0.3 annotated-types-0.7.0 antlr4-python3-runtime-4.9.3 anyio-4.11.0 audioop-lts-0.2.2 audioread-3.1.0 av-16.0.1 brotli-1.1.0 certifi-2025.10.5 cffi-2.0.0 charset_normalizer-3.4.4 click-8.3.0 cloudpickle-3.1.1 colorama-0.4.6 colorlog-6.10.1 cython-3.1.6 decorator-5.2.1 einops-0.8.1 ffmpy-0.6.4 filelock-3.19.1 ftfy-6.3.1 gitdb-4.0.12 gitpython-3.1.45 groovy-0.1.2 grpcio-1.76.0 h11-0.16.0 hf-xet-1.2.0 httpcore-1.0.9 httpx-0.28.1 huggingface-hub-0.36.0 hydra-colorlog-1.2.0 hydra-core-1.3.2 idna-3.11 importlib_metadata-8.7.0 joblib-1.5.2 jsonschema-5.3.0 lazy_loader-0.4 llvmlite-0.45.1 markdown-3.9 markdown-it-py-4.0.0 mdurl-0.1.2 mmaudio-1.0.0 msgpack-1.1.2 nitrous-ema-0.0.1 numpy-2.0.2 omegaconf-2.3.0 open-clip-torch-3.2.0 opencv-python-4.12.0.88 orjson-3.11.4 packaging-25.0 pandas-2.3.3 platformdirs-4.5.0 pooch-1.8.2 protobuf-6.33.0 pycparser-2.23 pydantic-2.11.10 pydantic-core-2.33.2 pydub-0.25.1 pygments-2.19.2 python-dateutil-2.9.0.post0 python-dotenv-1.2.1 python-multipart-0.0.20 pytz-2025.2 pyvers-0.1.0 pyyaml-6.0.3 regex-2025.10.23 requests-2.32.5 rich-14.2.0 ruff-0.14.2 safetensors-0.6.2 scikit-learn-1.7.2 scipy-1.16.2 shellingham-1.5.4 six-1.17.0 smmap-5.0.2 sniffio-1.3.1 soundfile-0.13.1 soxr-1.0.0 standard-aifc-3.13.0 standard-chunk-3.13.0 standard-sunau-3.13.0 starlette-0.49.0 tensorboard-2.20.0 tensorboard-data-server-0.7.2 tensordict-0.10.0 threadpoolctl-3.6.0 tomlkit-0.13.3 tokenizers-0.22.1 torch-2.7.1+cu118 torchaudio-2.7.1+cu118 torchvision-0.22.1+cu118 tqdm-4.67.1 typer-0.20.0 typer-slim-0.20.0 typing-inspection-0.4.2 typing-extensions-4.15.0 tzdata-2025.2 urllib3-2.5.0 uvicorn-0.38.0 wcwidth-0.2.14 websockets-15.0.1 werkzeug-3.1.3 zipp-3.23.0
```