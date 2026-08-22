# SFT-Conflict

SFT-Conflict 研究监督微调（SFT）造成的通用能力冲突，并尝试通过约束 Unembedding Laguerre-Voronoi Diagram（LVD）决策边际附近的 activation feature interference，降低微调过程中的通用能力损失。

完整工程需求与实验规划见 [REQUIREMENTS.md](REQUIREMENTS.md)。

## 服务器环境

当前环境已在以下硬件上验证：

- 4 x NVIDIA A100-SXM4-40GB
- NVIDIA Driver 535.129.03
- PyTorch CUDA 12.1 runtime
- 默认训练精度：BF16

训练与评测使用两个相互隔离的 Python 虚拟环境。虚拟环境、模型、数据、缓存和实验产物均位于 Git 仓库外的 `/data1/chenhang/SFT_Conflict/artifacts/`，不会同步到 GitHub。

## 训练环境

环境路径：

```text
/data1/chenhang/SFT_Conflict/artifacts/envs/train
```

激活环境：

```bash
source /data1/chenhang/SFT_Conflict/artifacts/env-manifests/activate-train.sh
```

退出环境：

```bash
deactivate
```

核心版本：

| Package | Version |
|---|---:|
| Python | 3.10.12 |
| PyTorch | 2.5.1+cu121 |
| Transformers | 4.48.3 |
| Accelerate | 1.3.0 |
| PEFT | 0.14.0 |
| Datasets | 3.2.0 |
| huggingface-hub | 0.28.1 |

训练环境已经通过以下检查：

- 4 张 A100 GPU 识别与 BF16 运算；
- Tiny Qwen2 standard LoRA 前向与反向传播；
- 两进程 Accelerate/NCCL all-reduce；
- `pip check` 依赖一致性。

## 评测环境

环境路径：

```text
/data1/chenhang/SFT_Conflict/artifacts/envs/lm-eval
```

激活环境：

```bash
source /data1/chenhang/SFT_Conflict/artifacts/env-manifests/activate-lm-eval.sh
```

退出环境：

```bash
deactivate
```

评测环境使用与训练环境一致的 PyTorch 和 Hugging Face 核心版本，并额外固定：

| Package | Version |
|---|---:|
| lm-eval | 0.4.9 |

评测环境已经通过以下检查：

- `lm_eval.evaluator.simple_evaluate` API 导入；
- `python -m lm_eval --help` CLI 启动；
- 首批通用能力任务注册检查；
- 4 张 A100 GPU 与 BF16 可用性；
- `pip check` 依赖一致性。

## Hugging Face 离线缓存

两个激活脚本都会设置以下离线变量：

```text
HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

统一缓存目录为：

```text
/data1/chenhang/SFT_Conflict/artifacts/hf_cache
```

当前服务器不在训练或评测过程中直接访问 Hugging Face。运行真实模型前，需要先将固定 revision 的模型和数据下载或迁移到该缓存目录；缺少缓存时程序应立即失败，不应在训练过程中临时联网下载。

## 环境清单

完整冻结依赖和本机使用说明位于：

```text
/data1/chenhang/SFT_Conflict/artifacts/env-manifests/
```

其中包括：

- `train-requirements.lock`
- `lm-eval-requirements.lock`
- `activate-train.sh`
- `activate-lm-eval.sh`

这些文件描述当前服务器上的可复现环境，但不包含模型权重、数据集或访问令牌。
