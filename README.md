# SFT-Conflict

SFT-Conflict 研究监督微调（SFT）造成的通用能力冲突，并尝试通过约束 Unembedding Laguerre-Voronoi Diagram（LVD）决策边际附近的 activation feature interference，降低微调过程中的通用能力损失。

完整工程需求与实验规划见 [REQUIREMENTS.md](REQUIREMENTS.md)。

## 当前实验：Base 自适应容忍带 Balance（2026-10-03）

[run.py](run.py) 已切换到 [band 配置](configs/method/anchor_ce_geometry_band.yaml)，
方法名仍是 `target_plus_anchor_ce_geometry`，配置目标为 `base_band_v2`。
保留 Full FT、Target CE 与权重0.2的 Anchor CE；用一个带边界限制的容忍带 loss
替代原来要求各方向漂移相等的 centered balance。

不再固定 `bmin=0.0025`、`bmax=0.01`：每个位置用 Base top128 的归一化 margin
中位数确定尺度，再按相对最强非 gold token 的 Base 概率比例缩放各方向宽度。
宽度比不再固定为4；频率只作弱修正。共同改善可被吸收，带内非均匀变化不受惩罚，
越界部分仍在同一个 loss 中受罚，不另加 guard loss。

保持 `ACTION="train"`、`TRAINING_MODE="full_ft"`、`RESUME_CHECKPOINT=None`，依次点击：

1. `METHOD="prepare_anchor_geometry_reference"`：校验/复用原有匹配的 Base 缓存，缺失时才记录。
2. `METHOD="target_plus_anchor_ce_geometry"`：50步 CE 热身、16组梯度 RMS 重新校准、20步渐增。

当前文件保持第2步的训练选择，未自动启动完整训练。已有兼容缓存可直接使用；
旧 centered checkpoint 与 λ 不可用于新目标的恢复。新自动运行名包含 `base_band_v2`。
模型、数据、FP32、FSDP、学习率及 `expandable_segments:True` 保持原设置。

已完成两卡 Tiny Qwen2 的准备、8步训练、导出和确定性模式下逐位一致恢复；
实际 Mistral head/activation 的9位置、512方向局部检查也通过。
这些验证不代表完整 Mistral 训练已经完成或质量优于 CE。
公式、边界行为、梯度实现与验证范围见 [新流程说明](docs/anchor-ce-band-20261003.md)。

2026-10-05：Qwen2.5-7B + GSM8K 的第50步校准 OOM 已定位到稀疏/稠密梯度相加时
额外分配整张输出头；仅将几何梯度改为 sparse 不足以修复。现在用共享 backward 将
几何行梯度直接累加到同一份输出头梯度，保留原训练设置。四卡 FP32 原配置已通过
104步、完整16组校准和20步渐增、checkpoint100、导出及全部1319条 GSM8K 测试集评估。
峰值已分配显存约37.01 GiB；183项相关回归通过。具体复现、数值及速度比较见
[Qwen OOM 记录](docs/anchor-ce-band-20261003.md#qwen25-7b-step-50-oom-investigation-2026-10-05)。
这仍不等于完整三轮训练已经验证。

## 历史快照：Anchor CE + 动态几何（2026-10-02）

> 下文描述旧 centered balance；当前点击入口已切换到上面的 `base_band_v2`。

当前方法是 `target_plus_anchor_ce_geometry`：
$L=L_{Target\,CE}+0.2L_{Anchor\,CE}+\lambda_t L_{geometry}$。
使用 Full FT，head 与 hidden 都参与当前边界及归一化范数的梯度；几何项约束方向间不均衡的
归一化 margin 漂移，不再使用 S0、d0、pressure 外层矩、winner risk 或 SO/SR 记录。

在 [run.py](run.py) 中设 `ACTION="train"`、`TRAINING_MODE="full_ft"`、
`RESUME_CHECKPOINT=None`，依次选择并点击：

1. `METHOD="prepare_anchor_geometry_reference"`：只读 Base，记录 Anchor 输出参照；首张 GPU 单进程，无 optimizer 更新。
2. `METHOD="target_plus_anchor_ce_geometry"`：前50步只用两项 CE，然后自动以加权 Anchor CE 梯度做16组 RMS 校准，20步渐增几何项。

两次点击保持模型、Anchor、tokenizer、长度与精度相同；参数见
[新配置](configs/method/anchor_ce_geometry.yaml)。热身计入原训练计划，scheduler 不重启。
当前文件中的 `native-evaluate`、Full FT CE 评估 checkpoint 等个人设置已保留，未自动切换或启动长训。

旧 `record_s0`、尺度/系数记录与 `target_plus_anchor_lvd+` 已退出当前点击入口；历史配置、CLI、
损失代码和实验产物保留兼容读取，旧系数/checkpoint 不得作为新方法的训练恢复来源。
新方法的 FSDP 使用 `use_orig_params=True`；活跃 head 的可微副本避免重计算读取已释放的分片视图。

已验证真实 Tiny Qwen2 的 CPU 与两卡 FSDP 准备、热身、校准、6步训练和逐位一致的恢复。
最新修复已撤回根 FSDP head 改动，恢复独立 head 分片；balance 采用一份全 head 梯度缓冲区的
分块局部求导，数学目标与一阶梯度不变。真实 Mistral 按当前四卡 FP32/原数据配置完成84步、
50步后的16×4校准、20步渐增、导出与评估，跨过原50/74步的崩溃点；1443步 scheduler 未缩短。
完整 CPU 回归：**6726 passed，19 warnings**；全项目 Ruff lint 与本次修改文件格式检查通过。
**这仍不是 Mistral 完整三轮训练或质量优于 Anchor CE 的证明。**
后续第113步崩溃已通过 GDB 定位为 PyTorch2.5.1 缓存释放的 use-after-free：
当前入口已移除 `max_split_size_mb:128`，改为 `expandable_segments:True`，禁用缺陷分支。
真实 Mistral 从原 checkpoint100 恢复到200步、保存和评估通过；loss、FSDP 与校准系数未改。
本次188项相关回归通过，原始 checkpoint 保留。原生栈与恢复产物见下方说明。
公式、权重、缓存身份、恢复协议与验证命令见
[动态几何说明](docs/anchor-ce-geometry-20261002.md)。

## 历史快照：LVD+ 混合最危险边界风险（2026-09-30）

> 下文“当前／新”及点击步骤为当日历史口径；当前点击入口已替换为上面的动态几何流程。

已实现 $L=L_{Target\,CE}+\lambda[O_{gain1}/S_O+3R/S_R]$，其中
$R=E_i\operatorname{softplus}(\max_{k\ne j_i}(z_{ik}-z_{i,j_i}))$。
保留原 S0、冻结 head、Top512 pressure 与 gain1 outer；新增风险不受原 $s\le0$ gate 限制。
使用 [混合配置](configs/method/lvd_plus_hybrid.yaml)，旧 gain1/gain32 配置保持可选。

[run.py](run.py) 已设为 `ACTION="train"`、`METHOD="record_s0"`。保持其余配置和
`RESUME_CHECKPOINT=None`，依次修改 `METHOD` 并点击运行：

1. `record_s0`：校验/复用或记录 S0；当前 Mistral 已有兼容缓存。
2. `record_lvd_plus_scales`：fresh Base 无梯度记录 64×4 global micro 的 $S_O/S_R$。
3. `calibrate_lvd_plus_lambda`：使用已记录尺度，为完整混合目标做参数梯度 RMS 定标并记录 lambda。
4. `target_plus_anchor_lvd+`：验证并复用记录，从 Base 开始正式训练。

后三步须保持相同数据、模型、精度、batch、校准设置及 world size；无需 pilot/d0 或手动复制缓存路径。
记录绑定实际模型、S0、数据/张量计划和目标配置，损坏或不兼容时报错，不覆盖旧记录。

验证：**6663 项 CPU 测试通过**；真实两GPU tiny 的独立记录→校准→复用→3步训练及保存通过，
27 个导出张量全部有限、head 不变。**Mistral 四卡验证因其他用户GPU占用在尺度阶段主动中止，
没有发布该模型的混合尺度/系数或执行训练更新**，不能据 tiny 测试声称生产模型已验收。
所有本轮验证进程已退出；未启动完整训练，也没有新的质量改善结论。

完整操作顺序、公式、缓存身份、恢复约束和验证边界见 [混合 LVD+ 使用说明](docs/lvd-plus-hybrid-20260930.md)。

## 历史快照：LVD+ region 2 越界优先（2026-09-29）

> 以下是当日 gain32 实验；当前入口已选择上面的混合配置，以下系数不可混用。

已按同一 token 位置对齐高置信度错误与 Base/Anchor CE activation：实际评测的 final 模型中，
188 个错误置信度 >99% 的位置里，48 个在 Base 中正确，而其中 42 个换回正确 activation 后旧 LVD+ 反而更高。
这不是标量响应越界后下降，而是多个方向汇总与外层矩的目标排序问题。

新增独立 [越界优先配置](configs/method/lvd_plus_s0_crossing.yaml)，`post_crossing_slope_gain=32`，
只平滑加强 region 2 斜率，region 0/1、Top-512、s0 表、外层矩保持不变。
固定 activation 上 region-2 局部梯度份额从 12.9% 增至 95.9%，但大部分错误/正确排序反例仍存在，
**不能据此宣称质量修复或 MCC 提升**。

当日 [run.py](run.py) 的 LVD+ 配置路径选择此实验；当时 native-evaluate/target_only 设置和评测路径未改。
开始新训练须显式改为 `ACTION="train"`、`METHOD="target_plus_anchor_lvd+"`，保持 `RESUME_CHECKPOINT=None`。
同一 s0 表直接复用，lambda 必须从 Base 重新定标；不得跨 gain 恢复。
完整分析、例子、公式和验证范围见 [region 2 加强报告](docs/lvd-plus-crossing-20260929.md)。

工程验收已完成：**5530 项 CPU 测试通过**；真实四卡新 Base **64×4 RMS + 3 步训练 + 保存/导出**通过。
新定标 $\lambda=3.0426464614901245\times10^{-11}$，291 个导出 FP32 张量全部有限，冻结输出头精确不变。
**没有启动完整训练，没有新的 MCC 改善结论**；原 s0 表和旧模型保留。

## 历史快照：record_s0 → 原始 LVD+（2026-09-28）

> 本节验收结果与系数对应 gain 1；不适用于上面的越界优先实验。

当前点击入口以**全词表概率锚点 $s_{0,j}$** 替代 $W_j/2$。
在 [run.py](run.py) 中先设 `METHOD="record_s0"` 点击运行，再改为
`METHOD="target_plus_anchor_lvd+"` 点击。保持 `S0_CACHE_PATH=None`、`RESUME_CHECKPOINT=None`；
不需要 pilot 或 d0。第一步只加载冻结输出头、用首张 GPU，不读取 Target/Anchor。
第二步校验同一 s0 表并重新做 Base RMS，然后开始正常完整训练，不是自动三步 smoke。

记录的是 $\|h\|_2\le512$ 的全词表 softmax 近似最大点，负对数概率 FW-gap 容差 $10^{-4}$。
**不是**将无穷远当作有限最大值，也不保证点位于自己的 cell 内。
Mistral 实测 **32768/32768 点全部收敛**；无约束有限最大点 **0**，有分离证据的无穷远上确界 **32768**：
其中 32001 个上确界为 1，767 个完全重复权重的 token 上确界为 $1/767$。
保存点 **31851 个严格在 cell 内、767 个并列、150 个在 cell 外**；所有 cell 均已证明非空且无界。
外部点保留使用，不删除 token 或回退到旧中心。原分段函数、label-only Anchor 筛选和全 gold 监督不变。

当前设置见 [s0 优化配置](configs/method/s0.yaml) 与 [s0 版 LVD+ 配置](configs/method/lvd_plus_s0.yaml)。
全部概率/gap/cell/分离证据已独立 FP64 复算；真实四卡新 **64×4 RMS + 3 步训练 + 保存/导出**通过，
新 $\lambda=7.814411678490693\times10^{-10}$。291 个导出 FP32 张量全部有限，冻结头精确不变。
最终 CPU 全量回归 **5195 项通过**，包含独立审计程序的 66 项测试。
原始统计、数学限制、重复权重处理和验收范围见 [s0 实现报告](docs/lvd-plus-s0-20260928.md)。
**没有启动完整长训，也没有质量优于 CE 的结论。旧中心/其他 s0 表的 checkpoint 和定标值不能混用。**

## 历史快照：half-Unembedding LVD+（2026-09-27）

> 本节“当前／新”指 2026-09-27 的 $W_j/2$ 模式；复现须显式选择旧 LVD+ 配置，以下系数不适用于 s0。

当前精确方法为 `target_plus_anchor_lvd+`：以 **gold 标签中心 $c_j=W_j/2$** 而非 Base $h_0$ 为参考，
覆盖所有合法 shifted supervised labels（包括 EOS 和 Base 预测错误的位置），不需要 $d_0$、reference 模型或 pilot。
三段判定优先执行 $s\le0$ 零响应，再按真实边际区分未越界/越界；中心边际允许为负，不设正下限。
保留均匀 Top-512 压力的**未中心化二阶矩 Frobenius 平方**；方向响应不是最终梯度，也不是中心欧氏 L2 损失。
公式、负边际反例、CE 对比和使用边界见 [LVD+ 当前说明](docs/lvd-plus-20260927.md#L1)。

**直接点击：**打开 [run.py](run.py)，保持 `ACTION="train"`、`METHOD="target_plus_anchor_lvd+"`、
`ANCHOR_DATASET="smart_flan_mistral_7b_label_coverage_v1"`、`AUTO_PREPARE_CACHE=True`、`RESUME_CHECKPOINT=None`。
当前 Mistral/CoLA 冻结头配置会先在 CPU 准备/复用源序标签覆盖集，再在同一个未更新的 Base 上做 **64×4 RMS 定标**。
不用先生成 pilot 或记录 d0；`METHOD="filtering_anchor"` 当前也默认是 CPU 标签覆盖，旧准确率流程须显式选择 `legacy_quality_v1`。
[独立新配置](configs/method/lvd_plus.yaml#L1)的 $\beta=0.0005$ 只是**未调优起始比例**，不是固定 lambda；旧 RMS 报告也不能复用。
正常点击使用完整训练设置，不是自动三步 smoke；新流程不保证错误概率去集中、Anchor 保持、优于 CE 或 native 崩溃修复。

**验收完成（2026-09-28）：**当前 64 组配置全量测试 **4071 passed**；真实覆盖集为 **12110 条样本／27943 种 label token**。
Mistral 四卡 Base 64×4 定标、3 步训练及导出通过，$\lambda=0.00014204440435962117$；291 个导出张量全部有限，冻结头与 Base 完全一致。
两轮 32 组定标失败保留原记录；最终未放宽门槛或丢弃主导组。全部验证已结束，**未启动完整训练，也未证明优于 CE**。
下方旧方法说明、设置表及 9 月 25/26 日结果保留历史口径；LVD+ 以本节及新文档为准，旧方法与旧 Anchor/YAML 仍可显式使用。

## 历史快照：LVD 三段式（2026-09-26）

> 本节“当前／新”均指 2026-09-26 的旧 LVD 协议；以下公式与验收结果不适用于 LVD+。

当前主协议为 `response_mode=three_segment`：**安全核严格零响应 → 平滑起步的线性缓冲区 → 越界后指数响应**。
数值指数阈值后使用非零斜率的切线尾段，不新增几何类别，也不是硬屏障。
公式、可微性、真实梯度与点击步骤见[三段式 LVD 说明](docs/lvd-three-segment-20260926.md#L1)。
`calibration_statistic=rms` 保留所有零梯度组与尖峰；`target_gradient_ratio=0.0005` 现在指加权梯度比的 RMS，
**仅作未经新目标搜索的起始尺度**。须重新生成兼容 pilot，不能复用旧 median 定标。
新四卡 10 步 RMS pilot 已通过健康、独立验证与定标门控，$\lambda\approx4.28613\times10^{-9}$；
从 Base 自动复用的 3 步训练与检查点存盘验收也已完成。完整训练尚未开始，不能据此推断 CoLA 改善。
下文 2026-09-25 的四段式搜索、旧 pilot 与 lambda 均保留为历史证据，不代表新模式的质量结果。

## 服务器环境

当前环境已在以下硬件上验证：

- 4 x NVIDIA A100-SXM4-40GB
- NVIDIA Driver 535.129.03
- PyTorch CUDA 12.1 runtime
- 默认训练精度：FP32

训练、lm-eval 与 AlpacaEval judge 使用三个相互隔离的 Python 虚拟环境。虚拟环境、模型、
数据、缓存和实验产物均位于 Git 仓库外的 `/data1/chenhang/SFT_Conflict/artifacts/`，不会同步到
GitHub。

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

## AlpacaEval judge 环境

官方 AlpacaEval 使用独立环境，避免其较新的 OpenAI、Datasets 和科学计算依赖改变训练环境：

```bash
source /data1/chenhang/SFT_Conflict/artifacts/env-manifests/activate-alpaca-eval.sh
```

环境固定为官方 `alpaca-eval==0.6.6`，源码 commit
`f19c323d8309d0d6f306bd26597db44fc6c62d57`。本地生成 `model_outputs.json` 不需要 API；只有
显式运行 LLM judge 才需要在终端环境中提供 `OPENAI_API_KEY`。密钥不应写入 `run.py`、配置或
日志。

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
- `alpaca-eval-requirements.lock`
- `activate-train.sh`
- `activate-lm-eval.sh`
- `activate-alpaca-eval.sh`

这些文件描述当前服务器上的可复现环境，但不包含模型权重、数据集或访问令牌。

## 直接点击 Run

推荐入口是仓库根目录的 [run.py](run.py)。在 VS Code 中打开该文件，点击右上角
**Run Python File** 即可；不需要输入任何命令行参数。即使 VS Code 当前选中了系统 Python，
该文件也会自动切换到本项目固定的训练环境。

**2026-09-26 历史入口（仅旧 LVD）：**`ACTION="train"`、`METHOD="generate_pilot"`、`MODEL="mistral_7b"`，
Target 为 CoLA，Anchor 为 `smart_flan_mistral_7b`。先点击生成新三段式 pilot；全部门控通过并确认
正式训练预算后，再改 `METHOD="target_plus_anchor_lvd"` 点击。保持 `RESUME_CHECKPOINT=None`、
`LVD_CALIBRATION_CHECKPOINT=None`；旧四段式发布不可复用，同协议 Base h0/d0 缓存无需重录。
操作与证据边界以[三段式说明](docs/lvd-three-segment-20260926.md#L1)为准，不把普通点击当成有界实验。

**下一段为 2026-09-25 历史快照：其中“当前／最新／已完成”仅指当日旧四段式实验。**

**LVD 的健康门控 pilot、自动复用流程和六组 100 步搜索均已完成；当前研究默认值为
$\beta=0.0005$（`target_gradient_ratio`），不再是早期 40 步胜出的 0.05。**
归一化修复后的 `pilot10-v4` 已完成生成与发布；最新四卡自动 100 步验证使用 **beta 0.005**，
当前 **beta 0.0005** 的点击配置、数据绑定与 lambda 重缩放另有 CPU 验证。
**流程可用不等于质量通过**：所有 LVD 候选仍有明显 Anchor 遗忘，未总体超过 Anchor CE，
也未验证完整三 epoch。checkpoint 只检查文件／元数据，未实际恢复。
操作说明和不同证据的边界见[最终报告](docs/lvd-pilot-lambda-search-20260925.md#L1)。
所有有界实验已结束、GPU 已释放，没有自动启动长训。

入口当前值是工作区运行配置，不是不可变实验默认值。模型、数据和结果都位于
`/data1/chenhang/SFT_Conflict/artifacts/`。

日常只需修改 [run.py](run.py) 顶部 `PRIMARY SETTINGS` 的七项：

| 设置 | 当前可选值 |
|---|---|
| `ACTION` | `train`、`audit-model`、`evaluate`、`merge-lora`、`build-reference-cache`、`record-anchor-d0`、`native-evaluate`、`audit-data`、`benchmark`；两个 `test_*` METHOD 会自动覆盖 ACTION |
| `MODEL` | `tiny_qwen2`、`qwen25_0.5b`、`qwen25_3b`、`qwen25_7b`、`mistral_7b` |
| `TARGET_DATASET` | `tiny_gsm8k`、`tiny_sst2`、`tiny_instruction`、`gsm8k`、八个 GLUE 名称、`codealpaca`、`code_bagel_hermes`、`infinity_instruct_code`、`evol_instruct`、`logiqa2`；独立评测还可用 `humaneval`、`codexglue_python`、`codexglue_java`、`alpaca_eval` |
| `ANCHOR_DATASET` | `none`、`tiny_general`、`smart_flan`、`smart_flan_<model>` |
| `METHOD` | `generate_pilot`、`target_only`、`target_plus_anchor_ce`、`target_plus_anchor_kl`、`target_plus_anchor_lvd`、`target_plus_anchor_ce_lvd`、`target_plus_null_lvd`、`filtering_anchor`、`record_anchor_d0`、`test_interference`、`test_crossing` |
| `TRAINING_MODE` | `full_ft`、`body_ft_frozen_head`、`lora` |
| `EVAL_CHECKPOINT` | `test_interference`/`test_crossing` 必填；其他评测中 `None` 表示 Base |

`TASK-DEPENDENT OVERRIDES` 支持 `None` 自动选择。`None` 不是某个固定实验值，而是让
入口根据主设置解析最终值；工作区中的显式 Boolean 或正整数设置优先：

| 自动项 | `None` 时的规则 |
|---|---|
| `GRADIENT_CHECKPOINTING` | 仅 3B/7B 的 `train`/`benchmark` 自动开启；评测和 0.5B/Tiny 自动关闭 |
| `FSDP_SYNC_EACH_BATCH` | 仅 FSDP 训练自动开启；其余动作和并行策略自动关闭 |
| `FSDP_CPU_OFFLOAD` | FSDP 训练中实际参数量 >=6B 时开启；显式 `False` 保持参数、梯度和 AdamW 状态在 GPU，须先验证显存 |
| `FSDP_KEEP_LOW_PRECISION_PARAMS` | 仅 7B Full FT + FSDP + BF16 自动开启 |
| `EVALUATION_MAX_SAMPLES` | 正式训练与独立评测使用完整 split；`MAX_STEPS>0` 的训练 smoke 自动限制为 256 |
| `NATIVE_MAX_NEW_TOKENS` | AlpacaEval 为 2048，GSM8K 为 256，HumanEval 为 512，STS-B 为 8，分类/CodeXGLUE 为 1 |

每次启动会先打印 `Resolved task-dependent settings`；CPU offload 的自动值需等待模型参数审计，
实际解析结果会写入训练前预览。`CPU_THREADS_PER_PROCESS` 是多进程启动时每 rank 的正整数
CPU 线程数，默认 1，预览同时记录 OMP/MKL 环境。其余 `FIXED DEFAULTS` 保持文件中的当前值。
特别地，`DISTRIBUTED_STRATEGY` 和 `GPU_IDS` 仍保留
显式硬件控制：3B/7B 非 LoRA 训练需要 `fsdp` 和至少两张 GPU；这两个值不会被程序静默改写。

`target_only` 必须搭配 `ANCHOR_DATASET="none"`；其余训练、记录或检验方法必须选择
`tiny_general` 或 `smart_flan`。其中
`target_plus_null_lvd` 只验证 Anchor 隐藏状态捕获接口，辅助损失仍为零；正式方法使用
`target_plus_anchor_lvd`。`record_anchor_d0` 是非训练 METHOD，会让 `run.py` 自动执行单进程
d0 记录，只加载 MODEL 与 Anchor。

### 长训恢复与崩溃诊断

[run.py](run.py) 的 `RECOVERY_SAVE_STEPS` 当前为 100：每 100 个 **optimizer step**
增加一次可续训 checkpoint，仍保留 epoch 验证和 best-model 选择，不额外运行验证。
设为 `None` 可关闭；YAML 对应 `training.recovery_save_steps`，默认 None，旧配置不变。
保存仍包含模型、AdamW、scheduler、RNG、双流状态；`save_total_limit` 保护 best 与最近恢复点，
不是永久保留每个 epoch 的旧 checkpoint。当前 Mistral FP32 实测保存约 99 秒，按 25 秒/步
估算每 100 步保存增加约 4% 耗时。

续训时将 `RESUME_CHECKPOINT` 指向完整的 checkpoint-N **目录**，不能指向仅供评测的 final。
保留原模型、数据、四卡和训练超参数；epoch 数仍表示总训练轮数。默认 None 仍表示从 Base
开始，不会自动选实验目录中“最新”checkpoint。第二轮及之后中途恢复的 Accelerate 1.3
skip-loader epoch 重置兼容性问题已有回归修复。

Linux 预检默认开启 native fatal-signal 栈，`SFT_CONFLICT_NATIVE_STACKS=0` 可关闭。
它不增加正常训练的逐步 CUDA 同步，但也不会防止底层崩溃。历史第 808 步后的 SIGSEGV
具体底层原因仍未证实；恢复、性能与验证范围见
[诊断与恢复报告](docs/lvd-fsdp-segv-808-20260924.md)。9 月 25 日自动定标后、正式训练前的
[另一例原生崩溃](../artifacts/results/lambda-mistral-auto-r005-s100-v1-20260925/failure-analysis.json#L1)
单独记录；新默认流程避开其已观察到的模型生命周期路径，不代表这些原生崩溃已被修复。

### 模型专属 Anchor 过滤（历史准确率策略）

> 以下保留旧准确率过滤与集合覆盖说明；现在复现它须显式设 `FILTERING_POLICY="legacy_quality_v1"`，默认 CPU 标签覆盖见顶部 LVD+ 说明。

先对完整原始 SMART FLAN 执行一次模型专属过滤：

```python
MODEL = "qwen25_7b"
ANCHOR_DATASET = "smart_flan"
METHOD = "filtering_anchor"
FILTER_THRESHOLD = 0.8
FILTERING_BATCH_SIZE = 4
```

该动作忽略 `ANCHOR_MAX_SAMPLES`，四卡各自复制 Base 模型并按长度分桶推理完整 Anchor，且只在
受监督 response 位置计算 Unembedding。单 token 回答必须预测正确；多 token 回答的
teacher-forced next-token argmax accuracy 必须不低于 `FILTER_THRESHOLD`。

通过准确率过滤后，程序对每条样本的去重 response token 集合运行 lazy greedy set cover，再
反向删除冗余样本。结果覆盖所有合格样本中出现过的 token 类型，但集合覆盖是 NP-hard 问题，
因此 manifest 明确记录该结果不保证全局最少样本数。输出包括：

- `artifacts/filtered_anchor_datasets/smart_flan_qwen25_7b/train.jsonl`
- `manifest.json`
- `output-token-dict.json`

后续训练将 `ANCHOR_DATASET` 改为 `smart_flan_qwen25_7b`。模型名和固定 revision 必须与过滤时
一致；该 tokenized 数据集会完整加载，不再受原始 `ANCHOR_MAX_SAMPLES` 限制。

每条样本的 correctness score 与阈值无关，会缓存在
`artifacts/anchor_filter_scores/smart_flan_<model>/max-length-<N>/`。修改
`FILTER_THRESHOLD` 后再次运行会复用这些 scores，只在单进程中重新 tokenization 和执行集合
覆盖，并在新产物完整写入后原子替换旧数据集；无需再次加载模型或重复四卡推理。

### Native 评测后的通用能力评测

`LLM_EVALUATION` 只在 `ACTION="native-evaluate"` 时生效。开启后，程序先完成
`TARGET_DATASET` 的原生评测并写盘，再释放原生 evaluator 的 GPU 显存，最后在隔离的
lm-eval 0.4.9 环境中评测同一个 Base 模型或 `EVAL_CHECKPOINT`：

```python
ACTION = "native-evaluate"
TARGET_DATASET = "sst2"
EVAL_CHECKPOINT = "/path/to/checkpoint/final"  # None 表示 Base
LLM_EVALUATION = True
LLM_EVALUATION_LIMIT = None  # 调试时可设为 32；正式结果保持 None
```

默认 `LLM_EVALUATION_TASKS` 覆盖阅读理解、自然语言推断、常识、物理与科学推理、真实性、
MMLU 知识、GSM8K 数学、BBH 推理、IFEval 指令遵循和 LAMBADA 语言建模，共 17 个任务或
任务组；`mmlu`、`bbh`、`race` 会由 harness 展开为各自子任务。任务列表、batch size、
few-shot、limit 和 integrity check 均可在 [run.py](run.py) 顶部直接修改。通用评测产物写入
同一 run 的 `llm-evaluation/`，原生命题评测仍位于 `native-evaluation/`。

lm-eval bridge 固定使用 `artifacts/envs/lm-eval` 与 `artifacts/lm-harness`。开启
`AUTO_PREPARE_CACHE=True` 时会在加载模型前检查全部任务数据，并临时联网补齐
`artifacts/hf_cache`；正式评估仍强制使用 Hugging Face 离线模式。关闭自动准备且缓存不完整
时，程序会在加载模型前直接报出缺失数据。

正式 lm-eval 子进程的 stdout、stderr 和 tqdm 会实时转发到启动 `run.py` 的 Terminal，并同步
写入同一结果目录旁的 `llm-evaluation.stdout.log` 与 `llm-evaluation.stderr.log`。metadata
保存两个完整日志路径和各自最后 64 KiB，长时间评测不再把全部终端输出积压在父进程内存中。

### Weighted-pressure LVD 历史协议（2026-09-26 三段式）

> 本节“当前／新”指当日旧 LVD，仍使用其 d0/pilot；不是 2026-09-27 的 LVD+。

当前设置见[方法配置](configs/method/weighted_pressure_lvd.yaml#L36)；理论详见[三段式说明](docs/lvd-three-segment-20260926.md#L1)。

- $\gamma_0=0.8$：$\rho\le1$ 时 $g=g'=0$；$1<\rho<1.25$ 是缓冲区，首 10% 平滑起步，随后线性。
	平滑宽度 $\tau=0.025$，$g(1.25)=0.2375$、$g'(1.25)=1$；这些是响应值和斜率，不是最终隐藏梯度的大小。
- 越界指数用 $\mu_3=1$，重新锚定于缓冲终点；`mu4=2` 只作旧配置兼容，不参与新响应。
	$C=6$ 后从 $\rho=7.25$ 切线延拓，斜率 $e^6$，不是硬截平或硬屏障；极小 d0 的数值下限另有几何例外。
- 当前 Top-512、`adapted_weight=False` 使用归一化均匀权重 $1/512$，仍以加权压力的**未中心化二阶矩 Frobenius 平方**为正则。
- 新 `calibration_statistic=rms` 用 $\lambda=0.0005/\sqrt{\operatorname{mean}(r_i^2)}$，32 组各累积 4 个 microbatch。
	保留所有零值和尖峰，不删样本或填 epsilon；要求至少 8 个正组、半组 RMS 倍数≤3、单组能量占比≤0.75。
	中位数统计只作描述，旧 median 模式保持原门槛。这个起始 RMS 比例未做质量搜索，也不保证未来梯度范围。
- 新响应、`core_transition_fraction` 与 RMS 规则绑定 pilot 兼容键；须新生成 pilot，不续训旧四段式 checkpoint。
	健康／独立验证仍严格执行；验收范围见新说明，不把短期验收等同于新方法质量改善。

### Weighted-pressure LVD 历史记录（2026-09-25 四段式）

> 以下含 pilot 子节的“当前／最新／最终”均保留 2026-09-25 口径，数值与验收范围不改写。
> 旧 publication、lambda 和 GPU 验收不适用于当前三段式；当前使用说明见上节及新文档。

**最终状态（2026-09-25）：工作流与有界搜索 COMPLETE，质量证据 LIMITED。**
当前为 $\gamma_0=0.8$、归一化均匀权重、**beta 0.0005**。下表六组均已完成
100 个 optimizer step，保持完整 1443 步 scheduler、seed 42、FP32 冻结头和四卡设置；
各 rank 的 Target 顺序匹配，LVD／Anchor CE 的 Anchor 顺序也匹配，Target-only 无 Anchor 更新。

| 已完成的 100 步实验 | Target validation token CE ↓ | Anchor validation accuracy ↑ | cached Base-correct retention ↑ |
|---|---:|---:|---:|
| Target-only | 0.3809572364 | 0.1683825212 | 0.2240386112 |
| LVD **beta 0.0005（研究默认）** | **0.2142770980** | **0.5792724777** | **0.7580569827** |
| LVD beta 0.005 | 0.3764226371 | 0.5889956532 | 0.7683325549 |
| LVD beta 0.05 | 0.2129685782 | 0.5638297872 | 0.7403082672 |
| LVD beta 0.5 | 0.3859410455 | 0.5403797758 | 0.7108827651 |
| 同条件 Anchor CE，weight 0.2 | 0.2207177187 | 0.7260352322 | 0.9254242566 |

完整数值见[100 步汇总](../artifacts/results/lvd-lambda-search-20260925/comparison-100-step.json#L1)，
选择依据见[最终决策](../artifacts/results/lvd-lambda-search-20260925/selection-decision.json#L1)。
这是**事后、非预注册的折中选择**：beta 0.0005 的 Target CE 比最佳 LVD 高约 **0.6144%**，
在 1% 范围内；相对 0.05，它的 Anchor accuracy／retention／CE 更好，最大总 pre-clip
梯度尖峰更小。0.005 的 retention 略高，但 Target CE 明显更差；没有候选全面支配其他候选。

同一 Anchor validation 的 Base accuracy 为 **0.7350720659**。所选候选的 Anchor CE 为
**2.0418589373**，Anchor CE 对照为 **1.1966381274**。LVD 比本次 Target-only 保留更多，
但仍远未达到理想保护；相对 Anchor CE 仅 Target CE 略低，Anchor 表现明显更差。
**不能称为总体胜过 CE、统计唯一最优或可放心长训三 epoch。**
最终代码完整回归 **1951 项通过**。精确默认 beta 0.0005 与 v4 pilot 的自动复用组合
另通过真实四卡 3 步启动、验证和 checkpoint-3 保存，见
[最终默认 GPU 验收](../artifacts/results/lvd-lambda-search-20260925/final-default-gpu-verification.json#L1)。
这是启动／保存检查，不是重新验证了 100 步或三 epoch 的质量。
Target 指标是 855 样本／2216 token 的内部 validation token CE，不是 CoLA Matthews；
Anchor accuracy／cached retention 分母分别为 8742／6423。两份 validation 都参与选择，
不是未见最终测试集，也不是生成准确率或通用 benchmark 结果。

[历史 40 步汇总](../artifacts/results/lvd-lambda-search-20260925/comparison-summary.json#L1)
保持原样：0.05 当时优于另外两个 LVD 比例，但该选择已被最终 100 步决策替代。

Mistral/CoLA 原 LVD（gamma0=0.2、容量加权）首轮标签塌缩的
[9 月 24 日质量审计](docs/lvd-quality-audit-20260924.md#L1)及旧实验产物保持历史原样，
不要仅凭训练 loss 下降继续长训。

LVD 超参数集中在带完整注释的
[configs/method/weighted_pressure_lvd.yaml](configs/method/weighted_pressure_lvd.yaml)，包括竞争方向数 K、
$\gamma_0,\mu_3,\mu_4,p,\lambda$ 策略、d0 位置/过滤、数值阈值、四段响应、容量权重、
pressure/moment 消融与 warmup。默认主协议选择**当前 Unembedding logits 最高的 512 个非参考 Token**，
$\gamma_0=0.8$、$\mu_3=1$、$\mu_4=2$、默认关闭容量权重、未中心化二阶矩，以及 target-only pilot
梯度比校准；没有跨模型固定 $\lambda$。

- [run.py](run.py) 顶部 `adapted_weight=False` 显式使用归一化均匀权重 $\alpha=1/K$；
	改为 `True` 恢复逆边际容量权重（$p=1$）。该入口开关覆盖 YAML `use_capacity_weights`；
	CLI 不指定 `--lvd-use-capacity-weights` / `--no-lvd-use-capacity-weights` 时保留 YAML 设置。
- $\gamma_0=0.8$ 表示允许侵蚀 80% 原始边际后才进入指数段，此时仍剩 20% 边际；
	核心内的正向侵蚀仍有线性惩罚，并非零惩罚区。决策边界的侵蚀比阈值变为 1.25，
	$g(1.25)=e^{0.25}\approx1.284$，仍不是硬屏障。
- 改变 gamma0 或容量权重可复用相同 Base 的 h0/d0 缓存，但需按新损失重新选择/定标 lambda，
	重新生成与正式配置兼容的 pilot，不能续跑旧目标的 checkpoint。入口已不再固定指向旧
	100 步 pilot；`LVD_CALIBRATION_CHECKPOINT=None` 只自动发现兼容且通过当前发布门槛的 pilot。
- [run.py](run.py) 顶部 `LVD_DIRECTION=512` 直接控制竞争方向数；实际数量为
	`min(LVD_DIRECTION, vocabulary_size - 1)`。每次 forward 重新排名，参考 token 是缓存的 Base argmax，
	排名不参与求导，hidden → pressure 的梯度保留。发生 argmax 翻转时，当前赢家必在竞争集合中。
	显式 YAML `neighbor_strategy: random` 仍可复现旧消融。
- `LVD_DYNAMIC_POSITIONS=True` 默认检查所有有效 response 预测位置（因果 shift 后，排除
	prompt、padding、EOS），数量随样本长度变化。`False` 则由 `ANCHOR_POSITIONS_PER_SAMPLE`
	控制固定采样上限。默认 `base_correct` 仍只保护 Base 原本预测正确的位置，因此
	`lvd/cached_positions` 可能小于 `lvd/requested_positions`；差值由 `lvd/uncached_positions` 显示。
- `position_chunk_size=4` 只是几何计算的分块大小，**不是每个样本只检查四个 token**。
	反向重算每块方向张量，所有 pressure 合并后再算全局二阶矩，不平均分块 loss。
- 新消融 `METHOD="target_plus_anchor_ce_lvd"` 使用
	$L=L_{target\ CE}+\text{ANCHOR\_WEIGHT}\,L_{anchor\ CE}+\lambda L_{LVD}$。
	Anchor CE 与 LVD 共享同一个 Anchor batch，FSDP 仍为单次合并 forward 和 summed backward。
	校准只比较 LVD 与 Target CE 梯度，不含 Anchor CE；当前 YAML 请求的中位比为 **0.0005**，
	来自本次 `target_plus_anchor_lvd` 的 100 步折中选择，不是每组或未来训练的上界，
	也不是联合方法已优于 CE 的证据。

训练中新增 `anchor/teacher_forced_accuracy`、`anchor/base_top1_agreement_cached` 和
`anchor/base_correct_retention`，均公开实际计数/分母。region0–3、crossing、flip 按所有
rank/microbatch 的实际边界或位置数汇总，不再等权平均局部比例。teacher-forced accuracy
覆盖全部监督 next-token；cached 保留率只覆盖缓存位置，二者不能混作生成准确率。
另有 active 指数截断、容量有效邻居数及 hidden/reference 范数监测。

当前显式分组定标报告为 `calibration_version=3`，沿用 v2 与训练一致的分布式梯度补偿，
另记录每 rank 的样本计划、分组和 hash。`calibration_batches=32` 表示 **32 个梯度比估计**，
每个估计累积 `calibration_accumulation_steps=4` 个原始 microbatch loss 的平均梯度，
不是合并 pressure 后另算大 batch loss，也不是先平均梯度范数。
四 rank、每流每 microbatch 一条样本时，共使用 $32\times4\times4=512$ 条唯一 Anchor
定标样本。旧已解析/fixed lambda 不会自动重缩放。
可选 `four_segment_linear_tail` 恢复 cap 后非零梯度；`asinh_weighted_sum` 是明确不同于
PDF 的实验性压强压缩目标。两者都不是严格硬屏障，后者未全面超过CE，不自动启用。

**从旧 random-K / 四位置实验迁移：**保留 `RESUME_CHECKPOINT=None`、
`ANCHOR_D0_CACHE_PATH=None`，先用 `METHOD="record_anchor_d0"` 点击记录动态位置缓存，
再按下面流程生成 pilot，最后选择两种 LVD METHOD 之一正式训练。记录与训练读取相同的 LVD 位置设置；
旧四位置缓存不能覆盖动态位置，会在加载训练模型前拒绝。完整全词表 d0 可复用于不同 K/排名策略，
但位置改变需重建，K/排名/位置改变均需重新校准 lambda，不能直接续跑旧 checkpoint。
40,000 条 Anchor 的全位置缓存可能很大（FP32 Mistral 每个保留位置约 144 KiB，含 h0 与 d0），
记录前应预算磁盘与时间；本次代码升级不会自动启动这项长任务。

#### 健康门控 pilot：先生成，再点击 LVD

流程已可用，以下是操作契约，**不是无条件长训建议**。当前 `MAX_STEPS=-1` 会保留
完整 epoch 计划，不会自动在 100 步停止；`RECOVERY_SAVE_STEPS=100` 也不是质量门控。
本次已完成实验使用独立的有界停止，不能与普通点击训练的预算混同。

| 设置 | 当前行为 |
|---|---|
| `PILOT_METHOD` | `target_plus_anchor_lvd`，或联合方法 `target_plus_anchor_ce_lvd`；须与后续正式 METHOD 一致 |
| `PILOT_STEPS` | 10 个 optimizer step，不是 scheduler 总长度或定标组数 |
| `PILOT_HEALTH_INTERVAL` | 每 5 步检查 Anchor health，终点另检查独立 Anchor validation |
| `target_gradient_ratio` | **0.0005**；按已验证 raw ratio 中位数重缩放 lambda，不是通用 fixed lambda |
| `calibration_batches` / `calibration_accumulation_steps` | 生成阶段测量 32 个梯度比，每个估计累积 4 个原始 microbatch loss |
| `TRAINING_MODE` | 原样保留 `body_ft_frozen_head` 或 `lora`；`full_ft` 直接拒绝 |
| `LVD_CALIBRATION_CHECKPOINT` | `None` 自动验证 exact-key publication v2 及所需数据指纹，复用其测量；不加载 pilot 模型 |

1. 保持正式的模型、Target、Anchor、训练模式、优化器、world size 和 LVD 设置，
	`RESUME_CHECKPOINT=None`、`MAX_STEPS=-1`。先备齐离线模型/数据及匹配的 **Base h0/d0**
	缓存；缺少或不匹配的 d0 不能靠自动发现补齐，需先点击 `METHOD="record_anchor_d0"`。
2. 设 `METHOD="generate_pilot"` 后点击 Run。pilot 的优化目标是 Target-only，
	但 Anchor 用于健康检查、独立验证和定标。保留 1443 步总计划、87 步 LR warmup，
	在 `PILOT_STEPS` 停止；普通 Target epoch／训练后评测关闭，Anchor 门槛仍启用。
3. 在生成进程中测量分组梯度；只有 health、validation、稳定性全部通过才导出并发布。
	报告记录实际 Target/Anchor/cache 指纹、raw norms/ratios 和分组计划。
	`RUN_NAME=None` 使用新目录；已有 pilot 输出目录会被拒绝。
4. 确认后续训练预算后，把 `METHOD` 改为同一个 `PILOT_METHOD`，保持
	`LVD_CALIBRATION_CHECKPOINT=None` 再点击。程序重验 publication UUID/key/report SHA、
	当前 d0 manifest 的 canonical-JSON SHA 和 raw 稳定性，取
	$\lambda=0.0005/\operatorname{median}(r_g)$；不加载额外 pilot 或重做定标反向。
5. 正式进程只加载／包装一个 **fresh Base FSDP 模型**。实际完整 Target/Anchor 指纹
	须在切出 Anchor training view 前匹配；只传递 lambda，不继承 pilot 权重、optimizer
	或 scheduler。缺失、不匹配或旧发布证据直接拒绝，不偷偷回退或自动续训。

**当前可复用 pilot 为归一化修复后的 `pilot10-v4`，再生成已完成。**它的 health retention
为 **0.9923776019**、validation retention 为 **0.9919040947**，32×4 分组稳定性通过。
publication UUID 为 `fcf3c4f5-aa28-45f8-b9e9-f8ace8dee6cb`，key 为
`88b03d86c21ad65412f03eac03038372d04c5914660bb9ea2ccb446ca592bd89`。
raw median **2.048945281912129e-5** 对应当前 $\lambda=24.402799060274905$。
原报告测量时的 beta 为 **0.05**；beta 不进入 compatibility key，故仅改该值可复用测量，
不是继承 pilot 模型，也不改写原报告。模型、数据、几何、优化器／scheduler、world size
或其他绑定设置改变时，仍须重新生成匹配 pilot。

[当前点击默认值检查](../artifacts/results/lvd-lambda-search-20260925/final-default-validation.json#L1)
已验证 generate/train 的实际 key 一致、实际数据指纹与 d0 绑定、beta 0.0005 重缩放，
且未初始化 CUDA、未加载定标模型。实际 Target training 为 **7696** 条；完整 Anchor
**5224** 条按 seed 20260925 分为 health 128、validation 128、calibration 512、training
**4456**，四部分互不重叠。Target 定标 512 条来自 Target **训练集**，不是独立 holdout；
历史上与正式训练前 640 条重叠 37 条。内部 Target validation 另有 855 条。

health／validation 门槛为 Base-correct retention ≥0.99、绝对 accuracy 下降 ≤0.01，
终点 hidden displacement 超过 Base replay 噪声至少 $10^{-5}$；定标 raw ratio 的
$\log_{10}$ IQR ≤2、前后半组中位数倍数 ≤3、peak/median ≤200。五步 pilot 健康但定标
不稳，十步通过；这些是工程门槛与本任务经验，不是通用最优步数或未来保护保证。

**最新真实自动链** `pilot10-v4` → `auto-r0005-s100-v1` 使用 **beta 0.005**、
$\lambda=244.02799060274904$，完成四卡 100 步／每 rank 400 microbatches 并保存
checkpoint-100。[v4 验证结果](../artifacts/results/lvd-lambda-search-20260925/workflow-v4-verification.json#L1)
为 `passed_default_workflow`，但只核对 24 个必需文件及元数据；模型、AdamW、scheduler、
RNG 均未实际恢复，fresh Base 也只是健康计数／范数与来源证据，不是权重逐项相等证明。
**不要把它误写为最终 beta 0.0005 的 v4 GPU 运行。**该比例已完成的 100 步质量结果来自
较早 v3 定标的 `r00005-s100-v1`（名字不代表 0.00005），lambda **24.40280249111826**；
与当前 v4 重缩放仅差约 $1.406\times10^{-7}$ 相对量，不声明 bitwise 等价。

需要进一步做有界 beta 对照时，可使用[搜索脚本](scripts/search_lvd_lambda.py#L1-L35)的
`--stage train`、`--method lvd`、`--auto-calibration`、`--ratio`、唯一 `--label` 及
显式 `--steps 100`；脚本默认 beta 0.0005、默认停止步数 40，不是无限训练或自动网格搜索。
它用独立停止保留完整 scheduler；不要通过改 `MAX_STEPS` 冒充同一 schedule 的对照。

**Target-only 归一化修复与原生崩溃、质量结论分开。**
[归一化说明](docs/lvd-target-only-normalization-20260925.md#L1)解释 HF 曾跳过 mean-loss
所需 `/4`；`model_accepts_loss_kwargs=False` 已纠正，修复后的 v4 已生成。
Combined LVD 与分组定标原本就正确，**不做 lambda/4 或 lambda×4 修正**。
该旧说明中的再生成状态保留历史口径，以本页最终状态为准；不是遗忘已修复的证据。
显式 `LVD_CALIBRATION_CHECKPOINT` 及旧 unseeded 路线仍会加载／重定标 pilot，
不建议用来绕过缺失发布。默认复用避开了上文零步 SIGSEGV 的第二次 FSDP setup 路径，
**没有证明普遍修复 native crash**。旧产物及旧审计均不追改。

未显式设置 `RUN_NAME` 时，checkpoint/result 目录使用
`click-{model}-{target_dataset}-{method}-{distributed_strategy}-{timestamp}`。未显式设置
`ANCHOR_D0_CACHE_PATH` 时，d0 目录名同时包含 `target-{target_dataset}` 和
`anchor-{anchor_dataset}`。显式路径保持原样；已有旧命名 artifacts 不会自动移动或重命名。

Unembedding 与平面 LVD 胞腔的一致性可直接点击验证：

```python
ACTION = "audit-model"
METHOD = "target_only"  # record_anchor_d0 会覆盖 ACTION，因此验证时改回 target_only。
ANCHOR_DATASET = "none"
LVD_AUDIT_MAX_POSITIONS = 8
LVD_AUDIT_VOCABULARY_CHUNK_SIZE = 256
```

该动作确定性抽取真实输入中的有效 token 位置，捕获实际进入 Unembedding 的隐藏状态。对每个
位置，它独立比较线性 Unembedding 最大 logit token 与最小 Laguerre power distance 对应的
LVD 胞腔 token，并检查该点是否满足所属凸多面体的全部 $V-1$ 个平面半空间约束。只有 token
完全一致、无违规平面、模型最终输出 argmax 一致且不存在精确并列胞腔时才通过。完整报告写入
`artifacts/results/<run-name>/lvd-geometry-audit.json`；词表分块只控制峰值显存，不近似或抽样
检查范围。

`d0_cache_path=None` 时，记录与训练根据模型、Anchor、样本预算、位置策略、过滤和 seed
自动导向同一路径；不同 K 可复用同一缓存。训练前严格验证模型 revision、Anchor 指纹、
token position、词表大小、过滤与数值口径。实现不构造完整 LVD 或 $V\times V$ Token 对；FSDP 使用冻结 CPU Unembedding 快照
和精确全局二阶矩解析 backward。公式、诊断字段、分布式推导、真实 Qwen/SMART 证据和正式
实验边界见 [docs/m3-lvd-validation-report.md](docs/m3-lvd-validation-report.md)。

### SFT interference 与 crossing 检验

两项检验适用于 Full FT、冻结头主体微调、LoRA、Anchor CE/KL/LVD 等所有已保存 checkpoint；
实际训练方法从 checkpoint 的 `resolved-config.yaml` 读取并写入结果。

```python
# 检验 1：token 方向上的 activation interference。
METHOD = "test_interference"
ANCHOR_DATASET = "smart_flan"
EVAL_CHECKPOINT = "/path/to/trained/final"
ANCHOR_D0_CACHE_PATH = "/path/to/all-reference-schema-v5-d0"

# 检验 2：逐 response token 的预测越界；不读取 ANCHOR_D0_CACHE_PATH。
METHOD = "test_crossing"
ANCHOR_DATASET = "smart_flan"
EVAL_CHECKPOINT = "/path/to/trained/final"
```

`test_interference` 在 cache 记录的相同 `(sample_id, token_position)` 上计算
$\Delta h_n=h_{\theta,n}-h_{0,n}$。固定基座 Unembedding 的单位 token 方向
$\hat W_{0,k}=W_{0,k}/\lVert W_{0,k}\rVert_2$，主统计为朝 token $k$ 的平均正向移动：

$$
m_k^+=\frac1N\sum_n\max(0,\hat W_{0,k}^{\top}\Delta h_n).
$$

所有词表 token 按 $m_k^+$ 降序写入 `token-direction-movement.csv`，并生成带坐标轴的
`token-direction-movement.svg`。`interference-summary.json` 还保存 signed/absolute/RMS/max
统计、$\sum_km_k^+$、绝对移动总和及 Top 1%/5%/10% 集中度。正式方法比较必须使用同一基座
模型、同一 Anchor 数据、同一位置 cache；cache 必须以 `d0_filter_mode=all_reference` 记录。
若要覆盖每个 response token，将位置策略设为 `all_selected_tokens` 后重新记录 cache。

`test_crossing` 顺序加载基座模型和 checkpoint，在 Anchor 的每个受监督 response token 上直接
比较输出 argmax 与 label。终端 JSON 同时打印 SFT 前/后 mismatch、`base correct -> tuned wrong`
的新增越界、恢复、纯预测翻转、accuracy 和比例；该流程不构造 LVD，也不访问 d0 cache。

### GLUE 八任务协议

CoLA、SST-2、MRPC、STS-B、QQP、MNLI、QNLI 和 RTE 使用固定的
`nyu-mll/glue@bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c`。每项任务将官方 `train`
按 seed 0、标签分层（STS-B 按 0.5 分数区间分层）确定性拆为 90% 训练和 10% 内部验证；官方
`validation` 仅作最终测试，MNLI 使用 `validation_matched`。隐藏标签的官方 `test` 不参与
训练、选模或结果汇报。

七个分类任务使用自然语言 prompt、固定 verbalizer 和 response-only Causal LM CE。STS-B
输出一位小数，训练使用整数位/小数位数值 token 分布的连续值期望 MSE，不使用 token CE；
最终评测使用最多 8 token 的确定性数值生成。分类评测比较所有 verbalizer 的条件对数似然。
任务指标为 CoLA Matthews、SST-2/MNLI/QNLI/RTE Accuracy、MRPC/QQP Accuracy 与 F1 的
平均、STS-B Pearson 与 Spearman 的平均。

完整协议通过以下 [run.py](run.py) 设置启动：

```python
ACTION = "train"
TARGET_DATASET = "cola"  # 替换为任一 GLUE 名称
MAX_STEPS = -1
NUM_TRAIN_EPOCHS = 10.0
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
TARGET_MAX_SAMPLES = None
TARGET_VALIDATION_MAX_SAMPLES = None
EVALUATION_MAX_SAMPLES = None
```

内部验证每 epoch 执行并按最低 validation loss 恢复最佳 checkpoint；训练结束后，官方
validation 使用独立的 `test_*` 前缀评测，避免与选模数据混淆。正式任务指标使用
`ACTION="native-evaluate"`：`EVAL_CHECKPOINT=None` 评测 Base，设置 checkpoint 路径则评测
SFT 模型。

MergeLM 的 BERT/RoBERTa 参考最优 LR 与 decoder 建议搜索网格被分别记录，不能将 encoder
数值视作 Qwen/Mistral 已验证最优值。CoLA/MRPC/STS-B 建议搜索
`(1e-5, 2e-5, 5e-5)`；其余任务建议搜索 `(5e-6, 1e-5, 2e-5)`，最终值仍需只根据内部 10%
验证集选择。实现、模板、完整 split 数量和 smoke 证据见
[docs/glue-validation-report.md](docs/glue-validation-report.md)。

### Logic & Reasoning 协议

逻辑与复杂推理只注册两条训练到评测链：

| `TARGET_DATASET` | 训练数据 | 默认论文评测 |
|---|---|---|
| `evol_instruct` | WizardLM Evol-Instruct 70k 完整 70,000 条 | AlpacaEval 805 条 |
| `logiqa2` | LogiQA2 English MRC train 12,567 条，dev 1,569 条内部验证 | English MRC test 1,572 条 0-shot Accuracy |

Evol-Instruct 使用 WizardLM 官方单轮训练格式
`{instruction}\n\n### Response:{output}`，只监督 output。长样本将 prompt 与 response 分开
tokenize，最多保留 prompt 开头 256 tokens，并把剩余容量优先留给 response；唯一空 output
样本按官方脚本语义监督 EOS。训练示例：

```python
ACTION = "train"
TARGET_DATASET = "evol_instruct"
TARGET_MAX_SAMPLES = None
MAX_LENGTH = 512
```

DARE 并未重新训练 Evol-Instruct，而是评测已经发布的 WizardLM checkpoint。本项目只复现其
AlpacaEval 推理协议：固定对话 prompt、greedy decoding、`temperature=0`、`top_p=1`、最多
2048 new tokens。完整 805 条生成后写出官方四字段 `model_outputs.json`：

```python
ACTION = "native-evaluate"
TARGET_DATASET = "alpaca_eval"
EVAL_CHECKPOINT = "/path/to/evol/checkpoint/final"
NATIVE_MAX_NEW_TOKENS = None  # 自动 2048
ALPACA_EVAL_RUN_JUDGE = False
```

默认只生成输出与 `scoring-commands.sh`，不会消费 API 额度。显式设置
`ALPACA_EVAL_RUN_JUDGE=True` 后，`chatgpt_fn` 复现 DARE；
`weighted_alpaca_eval_gpt4_turbo` 则运行当前 AlpacaEval 2.0 length-controlled 协议。两者
judge、reference 和指标不同，结果不得混列为同一协议。

LogiQA2 只读取固定 commit 中 `MRC/train.txt`、`MRC/dev.txt`、`MRC/test.txt`，不会使用根目录
自动 builder，因为后者会把 English/Chinese MRC 与 NLI 文件混入同名 split。训练与评测统一
使用 `Passage / Question / A-D / Answer:` 0-shot prompt；训练监督正确选项字母，最终评测比较
四个完整候选的条件对数似然并报告 Accuracy。超长样本保留 prompt 首尾，使 passage 开头和
末尾 question/options 同时存在，并始终保留答案 token。

DiFT 原文使用 LogiCoT 20k 训练、LogiQA2 0-shot Accuracy 评测；本项目按当前实验选择改为
LogiQA2 自身 train/dev 训练，不能将这组训练结果称为 DiFT 的原始 LogiCoT 数据协议。三个
HF 数据仓库当前均为公开、非 gated；固定 revision、模板、judge 边界与真实验证证据见
[docs/logic-reasoning-validation-report.md](docs/logic-reasoning-validation-report.md)。

### Code Generation 协议

代码训练支持三个独立目标：

| `TARGET_DATASET` | 训练数据与协议 | 默认论文评测集 |
|---|---|---|
| `codealpaca` | 固定 revision 的完整 20,022 条；LoRI Alpaca instruction/input/response 模板 | HumanEval |
| `code_bagel_hermes` | 从完整 3,958,847 条以 seed 0 均匀随机采样 20,000 条；DiFT code instruction 语义 | CodeXGLUE Python |
| `infinity_instruct_code` | Infinity-Instruct 7M 中按 code source/ability 过滤后，以 seed 0 随机采样 20,000 条；DiFT query/response 模板 | CodeXGLUE Python |

三者均使用 response-only Causal LM CE。`MAX_LENGTH=512` 时，代码数据按 LoRI
`max_prompt_length=256, keep_start` 协议保留 prompt 前部，并优先保留 response 前部，避免长
instruction 吞掉全部监督 token。Prompt 与 response 按 LoRI 分别 tokenize 后再拼接 token
IDs，因此 BPE token 不会跨越监督边界。CodeAlpaca 的 LoRI 参考训练为 2 epochs；DiFT 对每种 reasoning
ability 使用 20,000 条并训练至收敛。本项目仍由 `NUM_TRAIN_EPOCHS` 显式决定实际 epoch。

训练示例：

```python
ACTION = "train"
TARGET_DATASET = "codealpaca"  # 或 code_bagel_hermes / infinity_instruct_code
ANCHOR_DATASET = "none"
METHOD = "target_only"
MAX_LENGTH = 512
TARGET_MAX_SAMPLES = None
NUM_TRAIN_EPOCHS = 2.0          # LoRI CodeAlpaca reference
```

`code_bagel_hermes` 与 `infinity_instruct_code` 在 `TARGET_MAX_SAMPLES=None` 时自动使用各自的
20,000 条论文预算。手工填写正整数可用于 smoke。Infinity-Instruct 是 gated 数据；首次使用前
必须在 Hugging Face 页面接受条款，并在服务器执行 `huggingface-cli login`。Token 不应写入
`run.py`、配置文件或日志。

多卡 Infinity 训练会在父进程启动 Accelerate 之前只执行一次 7M code filter，并持久化派生
Arrow cache；各 rank 只复用该 cache，不再分别扫描 7,449,106 行。当前固定过滤器得到
1,902,034 条 code rows，再从中固定随机选择 20,000 条。

CodeAlpaca 训练启动时也会加载 HumanEval 做 teacher-forced 评估。Qwen2.5 会将
HumanEval/81 的 prompt 结尾三引号与 response 开头换行合并为一个 token；现在复用安全的
空白桥接掩码：仅当 response 一侧全为空白时，将混合 token 归入 prompt 掩码，代码内容
继续监督。输入 token、原始文本和 native 生成协议不变，不跳过样本，也不放宽非空白
边界错误。Mistral 的全部164条及 Qwen 原本有效的163条输入与标签保持一致；
复现和四卡验证见 [代码数据验证记录](docs/code-generation-validation-report.md#qwen25-7b-startup-boundary-fix-2026-10-10)。

HumanEval 复现 LoRI：164 题，每题采样 20 个 completion，`temperature=0.2`、`top_p=0.95`，
报告无偏估计 `pass@1`、`pass@5`、`pass@10`。运行方式：

```python
ACTION = "native-evaluate"
TARGET_DATASET = "humaneval"
EVAL_CHECKPOINT = "/path/to/codealpaca/checkpoint/final"  # None 评测 Base
ALLOW_CODE_EXECUTION = True
HUMANEVAL_SAMPLES_PER_PROBLEM = 20
EVALUATION_BATCH_SIZE = 4  # 单次 generate 最多并发4条序列，每题仍累计20个样本
NATIVE_MAX_NEW_TOKENS = None  # 自动 512
```

HumanEval 生成采用单卡分批：`EVALUATION_BATCH_SIZE` 同时限制展开后的生成序列数，
不会再将“4题 × 每题20次”一次展开成80条并发序列。每个问题仍保留20个 completion，
温度、top-p、长度上限及 pass@1/5/10 定义不变。种子在每个问题批次内只设置一次，
各次生成继续推进随机状态；结果记录分批方式、种子和并发上限。新的分批布局可能改变
具体采样结果，不承诺与旧布局逐 token 一致。
`native-evaluate` 不使用训练的 FSDP/DDP；即使 `GPU_IDS` 列出多张卡，当前原生评估仍在
第一张可见 GPU 上运行。四题80份生成及512输入+512输出的真实单卡验证见
[单卡 HumanEval 记录](docs/code-generation-validation-report.md#bounded-single-gpu-humaneval-generation-2026-10-12)。

HumanEval 会执行模型生成的 Python。默认 `ALLOW_CODE_EXECUTION=False` 并立即拒绝；显式开启后
使用 Linux user/network/mount/PID namespaces、临时目录、isolated Python、CPU/内存/文件/
FD 限制和逐 completion 超时。该防护仍不等价于独立 VM/容器安全边界，正式运行应放在可丢弃
的隔离节点。

DiFT 论文将代码指标简称为 CodeXGLUE “pass rate”，但用户指定的
`google/code_x_glue_cc_code_completion_token` 官方任务是 next-token completion，官方指标是
排除 `<s>`、`</s>`、`<EOL>`、`<pad>` 后的 token accuracy，不执行代码。本项目准确报告
`token_accuracy`，并额外保存 `model_subtoken_accuracy` 诊断：

```python
ACTION = "native-evaluate"
TARGET_DATASET = "codexglue_python"  # 第二次可改为 codexglue_java
EVAL_CHECKPOINT = "/path/to/dift-code/checkpoint/final"
```

完整模板、revision、指标公式、真实 cache 数量与验证证据见
[docs/code-generation-validation-report.md](docs/code-generation-validation-report.md)。

3B/7B 在单进程点击运行中只允许 LoRA；Full FT 和主体全量微调需要多 GPU FSDP。7B 在线
Anchor KL 需要同时驻留第二个 7B 参考模型，因此单进程入口会拒绝该组合；7B 可先使用
Anchor CE，或在 M2 使用参考激活缓存。

[run.py](run.py) 会根据 `DISTRIBUTED_STRATEGY` 自动启动 Accelerate，无需手工输入
`accelerate launch`：

- `single`：`GPU_IDS` 必须只有一张卡，例如 `(2,)`；
- `ddp`：`GPU_IDS` 至少两张卡，适合 LoRA 或每卡可完整容纳模型的训练；
- `fsdp`：`GPU_IDS` 至少两张卡，使用 Full Shard；冻结头/LoRA 使用
	`use_orig_params=true`，统一可训练的 Full FT 使用内存更低的 flat parameters；
- `GPU_IDS=(1, 3)` 只允许使用物理卡 1 和 3；子进程内部会重映射为本地 rank 0 和 1；
- 父进程只准备一次 cache，并自动为所有 rank 生成相同 run name 和空闲通信端口；
- 训练前预览和 run metadata 会记录 strategy、world size、rank 以及
	`CUDA_VISIBLE_DEVICES`。

当前 Mistral-7B + GSM8K + SMART Anchor CE、4 卡 Full FT 的已验证设置是：

```python
MODEL = "mistral_7b"
TARGET_DATASET = "gsm8k"
ANCHOR_DATASET = "smart_flan"
METHOD = "target_plus_anchor_ce"
TRAINING_MODE = "full_ft"
MIXED_PRECISION = "bf16"
OPTIMIZER = "adamw_torch"
DISTRIBUTED_STRATEGY = "fsdp"
GPU_IDS = (0, 1, 2, 3)
LEARNING_RATE = 2e-5
MAX_LENGTH = 512
ANCHOR_MAX_SAMPLES = 40_000
GRADIENT_ACCUMULATION_STEPS = 4
FSDP_KEEP_LOW_PRECISION_PARAMS = True
FSDP_SYNC_EACH_BATCH = True
GRADIENT_CHECKPOINTING = True
```

该点击路径使用 PyTorch AdamW；FSDP 下禁止 bitsandbytes optimizer。

### 7B Full FT FSDP 显存与稳定性说明

Accelerate 1.3 的默认 BF16 FSDP 会把可训练 FlatParameter 强制升为 FP32，并让标准 AdamW
创建两份逐元素状态。Mistral-7B Full FT 在第一个 step 创建状态后，每 rank 接近 40GB，第二
个 backward 的 418 MiB unshard buffer 因容量不足而 OOM。降低序列长度不能解决该持久状态
问题。

早期实现同时在 Accelerate launcher 中请求 transformer wrap，却又在
`TrainingArguments.fsdp_config` 中传入 `min_num_params=100M`。Transformers 会将后者写回
环境并覆盖 launcher，导致 32 层 Attention 被聚合到一个巨大的 root FlatParameter；第一个
optimizer step 创建 Adam 状态后，第二步需要一次性 all-gather 约 2.50 GiB，从而在每卡已用
约 35.7 GiB 时 OOM。

当前 Full FT + Anchor CE 方案包含五项显式设置：输出 head 单独成为 FSDP unit；Target 与
Anchor 在 sequence 维度右侧补齐后沿 batch 维拼接，只执行一次 forward 和一次 backward；
FlatParameter 保持 BF16；optimizer 使用 PyTorch AdamW；CUDA AdamW 强制
`foreach=False`，不使用自动选择的 multi-tensor backend。AdamW 的一、二阶
状态随 FlatParameter 保持 BF16，避免标准 FP32 参数与两份 FP32 moment 超过 40GB 显存。
该方案仍是 AdamW，但不等价于 FP32-master/FP32-state AdamW。

旧实现每个 micro-batch 依次执行 Target backward 和 Anchor backward。PyTorch FSDP 2.5.1
源码明确说明 multiple backwards 的异步 reduce-scatter 可能乱序，需要额外竞态规避；该路径在
不同 rank、不同 step 多次出现 native `SIGSEGV`。当前 Anchor CE 单 pass 保持两项 loss 各自的
token 归一化和权重，Tiny 不等长 batch 的逐参数梯度与原联合目标一致。真实 Qwen2.5-7B 四卡
生产数据几何运行 1,300 step，越过此前最早的 step 1,276 崩溃点；峰值约 31.6GB/rank。
FSDP 多损失训练会将 Target 与 Anchor 在 batch 维拼接为一次 forward，并在内部拆分 logits
和 Anchor hidden states；CE 与 LVD 求和后执行一次 backward。>=6B 模型的自动策略仍启用 CPU
offload，但这不是 FP32 的必需条件。足够 GPU 显存下可显式设置 `FSDP_CPU_OFFLOAD=False`，
同时保留逐 microbatch 同步和原 accumulation：梯度由 FSDP 在 GPU 分片上累积，CPU 独立
累积器不再启用。该选择不降低参数/AdamW 精度，也不改变 LVD 位置、方向或有效 batch。
CPU/GPU 浮点归约和更新内核不保证逐 bit 一致，不能据此承诺完全相同的最终任务分数。
当前 Mistral CoLA 四卡 FP32 的受控试验及验证范围见[性能对照](docs/lvd-fsdp-performance-20260923.md)；
更换模型、GPU 数量、长度或 batch 后须重新验证训练、保存和评测峰值。

项目观察到两次无 Python traceback 的单-rank 原生 `SIGSEGV`：一次是已知不受支持的
`FSDP + adamw_bnb_8bit` 在 step 2,349；切换 PyTorch AdamW 后又在 step 2,706 发生一次。
服务器禁止 core dump，普通用户也无权读取 kernel journal，因此不能把第二次崩溃精确归因到
某个 C++/CUDA 符号。该 run 已确认 loss、参数、optimizer state 与 allocated memory 都正常，
且没有 checkpoint 活动；Anchor 总条目数不是直接 GPU 分配原因。

PyTorch 2.5.1 会在 CUDA 参数上将 `AdamW(foreach=None)` 自动路由到 multi-tensor foreach，
该路径需要额外约一个参数副本的峰值内存。失败 run 每 rank 虽只有约 11.5GB 活跃张量，allocator
却长期保留约 37GB。强制 `foreach=False` 后，两轮真实 Qwen2.5-7B 四卡 100-step 验证全部完成，
包括一轮保持正式 dataloader/epoch 几何的前 100 step；峰值 allocated 约 19.9GB，reserved 稳定
在约 26–31GB。由于原始 native 栈不可用，这是一项由对照内存证据支持的保守规避，而不是对
底层符号的绝对证明。`PYTHONFAULTHANDLER=1` 已为未来原生故障提供各线程 Python 栈。

点击入口将 SMART Anchor 限制为发布顺序中的前 40,000 条，以减少四个 rank 重复 tokenization
和 Python 常驻对象；这会改变 Anchor 样本预算，但不会改变每 step 的 batch size 或损失定义。

所有 FSDP 模式的 optimizer 状态按 rank 原样保存为
`optimizer.rank-NNNNN-of-NNNNN.pt`；schema-3 manifest 绑定 optimizer、training mode、
`use_orig_params`、world size 与全部分片，恢复训练必须保持相同配置和 GPU/world size，并兼容
既有 schema-1/2 Full-FT checkpoint。中间模型 checkpoint 使用对称的 FSDP
SHARDED_STATE_DICT，训练结束才聚合并同步发布标准 Hugging Face Safetensors。evaluation 后保存前，所有 rank 会先同步
CUDA、回收 Python 对象并释放 allocator cache；DCP 保存计划和完成状态通过专用 Gloo process
group 在 CPU 上交换，不再让默认 NCCL `gather_object` 为纯元数据分配 CUDA buffer。模型分片先
写入临时目录，只有生成 `.metadata` 后才原子发布；恢复时也使用 Gloo 并拒绝缺少 `.metadata`
的残缺 checkpoint。Torch 2.5.1 打印的 FSDP/ShardedTensor `FutureWarning` 是旧 state-dict API
的弃用提示，不表示 checkpoint 保存失败。

Qwen2.5-7B + SST-2 已在 4 张 A100 40GB 上完成 8-step checkpoint/恢复验证和上述两轮
100-step 对照验证；所有 loss 有限，四卡 volatile ECC 均为零。

`AUTO_PREPARE_CACHE=true` 时，入口会先在独立子进程下载选中资源的固定 commit，然后回到
严格离线训练；失效的本地 `127.0.0.1:1081` 代理会在该准备子进程中自动忽略。当前
Qwen2.5-7B commit `d149729398750b98c0af14eb82c78cfe92750796`、Mistral-7B v0.3 commit
`caa1feb0e54d415e2df31207e5f4e273e33509b1`、Qwen2.5-0.5B、GSM8K、八个 GLUE config、SMART
FLAN 和 HellaSwag 均已缓存。Qwen2.5-7B 与 Mistral-7B 都通过离线模型/LoRA/Unembedding
审计和真实反向传播。

每次微调开始前都会打印并保存 `preflight-preview.txt/json`，内容包括微调方式、可训练与
冻结参数、Unembedding/tied-weight 状态、数据集与损失/约束关系，以及随机 3 条 target
样本；配置 Anchor 时还会独立随机打印 3 条 Anchor 样本。两组样本都先经过正式
`CausalLMCollator`，再直接从 `input_ids`、`attention_mask` 和 `labels` tensor 反解文本；
不会从 DataLoader 取 preview batch，也不会直接解码原始数据记录。
控制台同时区分 `FULL INPUT TEXT` 和 `FIRST-LABEL PREDICTION PREFIX`：前者是标准
teacher-forcing 完整序列，因此包含 gold response；后者根据 `labels != -100` 找到第一个监督
位置并排除该 token，表示预测第一个 label token 时 Causal LM 实际可见的前缀。例如 SST-2
的完整序列可包含 `Sentiment: positive`，但预测 `positive` 的前缀严格停在 `Sentiment:`。
训练 loss 使用 `logits[:, :-1]` 对齐 `labels[:, 1:]`，因此 gold token 不会被用于预测自身。

Trainer 第一次实际评估前会以同样方式随机打印 3 条 evaluation tensor，后续 epoch/final
评估不重复打印。`native-evaluate` 也只打印一次：分类任务展示 gold candidate 对应的模型
输入和 label tensor；GSM8K/STS-B 展示实际 prompt 输入 tensor 与独立 gold label tensor。
Anchor 只参与训练约束而不属于 evaluation dataset，因此不会在每次评估前重复打印。所有
随机样本使用系统随机源，不受训练 seed 固定，因此不会每次选择相同的三个索引。

[src/sft_conflict/cli/main.py](src/sft_conflict/cli/main.py) 也支持无参数运行，默认执行一个
无需下载的 Tiny 实验；日常使用仍建议固定点击根目录的 [run.py](run.py)。

## M1 训练底座

第一步代码已经按以下主链实现：

```text
CLI + strict YAML dataclass
	-> offline model/tokenizer factory
	-> target + optional anchor datasets
	-> deterministic dual-stream scheduler
	-> SFTConflictTrainer (Transformers Trainer + Accelerate)
	-> loss plugins + checkpoint + local tracking
```

实现边界和 Future Localization 蓝本迁移关系见
[docs/m1-migration-matrix.md](docs/m1-migration-matrix.md)。参考快照来源与许可证记录见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和
[docs/reference-snapshots.json](docs/reference-snapshots.json)。参考目录不会加入运行时
`PYTHONPATH`。

M1 的真实验收矩阵、产物路径和已知第三方限制见
[docs/m1-validation-report.md](docs/m1-validation-report.md)。

## M2 数据、参考缓存与基准

M2 已加入统一 `DatasetRegistry`、逻辑样本协议、数据 manifest、Anchor 位置选择、分片参考
缓存、原生任务评测、泄漏审计和六组 benchmark 编排。实现与真实验证证据见
[docs/m2-validation-report.md](docs/m2-validation-report.md)。

所有主流程仍通过 [run.py](run.py) 顶部设置直接运行：

- `ACTION="build-reference-cache"`：将 `ANCHOR_DATASET` 设为 `tiny_general` 或
	`smart_flan`。默认按 seed 为每条样本选择 4 个 response Token，只保留基座 argmax 与真实
	next token 一致的位置，使用 FP32 Safetensors 分片。缓存支持哈希完整性校验、断点续生成和
	按 `sample_id` 随机读取。
- `ACTION="native-evaluate"`：`TARGET_DATASET` 使用八个 GLUE 任务之一、`gsm8k`、
	`logiqa2`、`alpaca_eval`、`humaneval`、`codexglue_python` 或 `codexglue_java`。GLUE
	分类任务使用候选回答条件对数似然，STS-B 使用数值生成，GSM8K 使用确定性生成与数值
	Exact Match。设置
	`EVAL_CHECKPOINT` 可评测 Full FT、冻结头或 LoRA checkpoint，不设置则评测 Base。
	`EVALUATION_BATCH_SIZE` 独立于训练 batch；运行时会打印样本数、batch 数和实时进度。
	`NATIVE_MAX_NEW_TOKENS` 是每道生成题的最大输出长度，不是整个评测的总长度。Mistral-7B
	在 A100 上以 batch 4、每题最多 256 token 评测完整 1,319 条 GSM8K，实测量级约 70--75
	分钟；结果文件在全部样本完成后原子写入。
- `ACTION="audit-data"`：对 target train、Anchor 和 target evaluation 执行 Unicode
	规范化精确哈希与近重复审计，保存 role-pair 报告。
- `ACTION="benchmark"`：固定生成 `base`、Full FT、冻结头主体微调、LoRA、Anchor CE、
	Anchor KL 六组计划。`BENCHMARK_EXECUTE=False` 只生成计划；设为 `True` 后按状态文件断点
	执行训练、原生目标评测与 pinned lm-eval，并输出二维 JSON/CSV 表。正式模式需设置
	`BENCHMARK_FORMAL=True` 和至少 3 个不同 seed，且禁止 evaluation limit。

Anchor 采样由 `ANCHOR_SAMPLING_STRATEGY` 控制：`task_balanced` 是主协议，
`mixture_proportional` 和 `random` 用于消融。每个训练微批的任务组成写入
`anchor-batches.rank-<rank>.jsonl`。固定 SMART 发布物只含 `prompt/response`，上游脚本已删除
task 元数据，因此未提供外部 sidecar 时应使用 `mixture_proportional`；manifest 会明确记录该
限制，而不会将所有样本伪装成可分层任务。权威 JSONL sidecar 可通过
`ANCHOR_METADATA_PATH` 指定，每行必须以 `sample_id` 或 `source_index` 为键。

`ANCHOR_MAX_SAMPLES=None` 会扫描完整 400,000 行 SMART train split；在 Qwen2.5-7B、长度 512
下得到 336,347 条有效 Anchor，另有 63,653 条因截断后没有监督 response token 而被记录并
过滤。SMART v3 response-only mask 支持仅由边界空白造成的跨界 BPE token，并继续对真正的
prompt/response 内容跨界 fail-fast。该版本变化进入数据指纹，不能从旧 SMART v2 checkpoint
断点恢复。加载时仅 rank 0 显示 `tokenizing`、`validating` 和 `fingerprinting` 三阶段进度；
tokenizing 条同时显示 scanned、accepted 与 filtered，避免四个 FSDP worker 重复刷屏。

参考缓存的 `REFERENCE_STORAGE_DTYPE` 默认为 `float32`。使用 `bfloat16` 或 `float16` 时，
必须通过预注册误差预算；每个分片会检查隐藏状态绝对误差、argmax 一致率、全词表边际绝对与
相对误差及边际符号翻转率，超阈值立即失败。

benchmark 的多卡训练条件由父进程逐项启动独立 Accelerate DDP/FSDP 子进程，训练完成后再
回到单进程执行目标与通用评测。当前 4×A100-40GB 工作流不会运行 7B exact online Anchor
KL，因为第二个完整 7B 参考模型会超出已验证的显存边界；完整六组矩阵应使用
`qwen25_0.5b`，或在更大显存环境重新验证该条件。

### 安装

激活训练环境后，以离线方式安装 editable package：

```bash
source /data1/chenhang/SFT_Conflict/artifacts/env-manifests/activate-train.sh
cd /data1/chenhang/SFT_Conflict/SFT-Conflict
python -m pip install --no-build-isolation --no-deps -e .
sft-conflict --help
```

### 配置

`configs/experiment/` 保留可独立解析的 M1 YAML 配置，作为实验快照和高级入口，覆盖：

- Full FT；
- 冻结 Unembedding 的主体微调；
- standard LoRA；
- Target + Anchor CE；
- Target + Anchor KL；
- 零权重 LVD plugin 双流路径。

运行前必须将 `REPLACE_WITH_*_COMMIT` 替换为本地 cache 中对应的不可变 revision。配置
loader 会拒绝未知字段以及 mask、WANDA/SNIP、解释性 LoRA、Sophia、probing 和 two-stage
alternating 等旧字段。

命令行也提供全部带默认值的统一 preset 入口：

```bash
sft-conflict run
```

其可选覆盖参数与 [run.py](run.py) 顶部设置一致，包括 `--model`、`--target-dataset`、
`--anchor-dataset`、`--method`、`--supervision`、`--training-mode`、`--mixed-precision`、
`--max-steps`、`--learning-rate`、最大长度、样本数、梯度累积、LoRA 参数、两个 batch
size、`--seed`、`--artifact-root`、cache 策略和 `--run-name`。所有参数均有默认值。

### 训练与恢复

模型结构和 step-0 Unembedding 审计：

```bash
sft-conflict audit-model \
	--config configs/experiment/qwen25_05b_sst2_body_ft_smoke.yaml
```

启动训练：

```bash
sft-conflict train \
	--config configs/experiment/qwen25_05b_sst2_body_ft_smoke.yaml
```

从 Trainer checkpoint 继续训练：

```bash
sft-conflict train \
	--config configs/experiment/qwen25_05b_sst2_body_ft_smoke.yaml \
	--resume-from-checkpoint \
	/data1/chenhang/SFT_Conflict/artifacts/checkpoints/<run_id>/checkpoint-<step>
```

只评测 checkpoint，并使用独立 run ID 保存结果：

```bash
sft-conflict evaluate \
	--config configs/experiment/qwen25_05b_sst2_body_ft_smoke.yaml \
	--run-name <eval_run_id> \
	--eval-checkpoint \
	/data1/chenhang/SFT_Conflict/artifacts/checkpoints/<run_id>/checkpoint-<step>
```

每次运行会保存解析后配置、Git 状态、依赖/GPU 信息、模型与参数审计、数据指纹、容量
检查、JSONL 指标和 TensorBoard event。`resume_from_checkpoint` 会验证双流数据指纹、seed、
batch size、比例、world size、epoch permutation 和 rank state；`eval_checkpoint` 是独立模式。

LoRA Adapter 的 merged 导出是独立命令，不会覆盖训练中的 Adapter 模型：

```bash
sft-conflict merge-lora \
	--config configs/experiment/qwen25_05b_gsm8k_lora_smoke.yaml \
	--adapter /data1/chenhang/SFT_Conflict/artifacts/checkpoints/<run_id>/final \
	--output /data1/chenhang/SFT_Conflict/artifacts/checkpoints/<merged_run_id>
```

导出前会在固定输入上比较 Adapter 与 merged 模型的 Logit 和 argmax，并保存
`merge-audit.json`；输出目录必须为空。

### 配置迁移与 lm-eval

旧 Future Localization 配置只能通过显式单向迁移命令转换，并生成丢弃字段报告：

```bash
sft-conflict migrate-config \
	--input <legacy.json-or-yaml> \
	--output <new.yaml> \
	--report <migration-report.json> \
	--model-revision <immutable-model-commit> \
	--dataset-revision <immutable-dataset-commit>
```

lm-eval 使用独立 0.4.9 环境，不导入训练进程：

```bash
sft-conflict lm-eval \
	--model-args pretrained=<local-model-path>,dtype=bfloat16 \
	--tasks hellaswag,gsm8k \
	--limit 16 \
	--output /data1/chenhang/SFT_Conflict/artifacts/evaluation_cache/smoke
```

桥接层会先通过固定 0.4.9 `TaskManager` 验证任务名，再保存完整命令、stdout/stderr 和退出
码，并校验请求的单任务或展开后的任务组被实际结果完整覆盖且指标有限。request cache 的路径
绑定 harness revision、任务集合、模型/Tokenizer 参数、few-shot 和 seed。

### 验证

CPU/Tiny 模型测试：

```bash
pytest -q
ruff check src tests
ruff format --check src tests
```

2026-09-25 最终完整 CPU suite 为 **1951 passed、9 warnings**，耗时 276.93 秒；
包含 **280 个 verifier cases**，全项目 Ruff lint 通过。
已知[verifier 测试](tests/unit/test_lvd_workflow_verifier.py#L1)仍有一处纯 EOF 格式警告，
不宣称全项目 format 全通过。最终精确默认已完成四卡 3 步启动与保存验证，所有有界作业结束。
这些工程检查不证明三 epoch 质量、最终 checkpoint 实际恢复或 native crash 根治。
Code 阶段的历史证据见 [docs/code-generation-validation-report.md](docs/code-generation-validation-report.md)。

两卡 DDP 和 FSDP Full Shard 回归入口位于
[tests/distributed/run_trainer_smoke.py](tests/distributed/run_trainer_smoke.py)。除 DDP/FSDP
基础冒烟外，FP32 no-offload 路径已完成六步周期保存/评测/重载及独立恢复；全部参数、
AdamW 状态、scheduler、两路数据状态和 RNG 的逐项比较通过。
[梯度累积 oracle](tests/distributed/run_fsdp_accumulation_oracle.py)另验证裁剪前后梯度及
最终参数与 dense 参考的一致性；真实模型长验证的状态见[性能对照](docs/lvd-fsdp-performance-20260923.md)。

M1 已完成 Qwen2.5-0.5B 的 SST-2/GSM8K × Full FT/LoRA 四组 20-step 真实冒烟、SMART
Anchor CE/KL/Null-LVD 三组 20-step 冒烟、两卡 DDP/FSDP 20-step 回归、Qwen2.5-7B
20-step LoRA、Mistral-7B 审计/反向传播和 Future Localization loss parity。正式多 seed
M2 已完成任务原生指标、参考激活 cache、泄漏审计和完整 benchmark 编排；正式三 seed、完整
九任务通用能力矩阵仍是实验运行工作，不属于代码缺口。

阶段边界如下：

- M1 已有：Target response-only CE、Anchor CE、在线冻结参考模型 KL、Full FT、冻结头主体
	微调、standard LoRA、双流训练和 Null-LVD 接口；
- M2 已有：完整 GLUE/GSM8K/SMART 数据闭环、稳定 Anchor 位置选择、参考激活缓存、原生
	评测、数据泄漏审计和可恢复 benchmark；
- M3 实现：全词表 d0 缓存、当前 logits Top-K LVD margin、动态位置、Anchor CE+LVD 消融、四段压强响应、容量权重、Unified-Cov/Centered-Cov 和真实
	LVD 保护损失；
- M4 实现：完整 baseline、实验矩阵、统计分析与论文图表。
