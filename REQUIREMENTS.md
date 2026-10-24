# SFT 特征干扰约束项目需求文档

## 1. 文档目标

本项目旨在构建一套可复现的 LLM 监督微调（SFT）实验框架，并实现一种基于 Unembedding Laguerre-Voronoi Diagram（LVD）决策边际和容量加权协方差的通用能力保护损失。

项目最终需要回答三个问题：

1. 目标任务 SFT 是否会造成通用能力下降，并伴随可测量的隐藏状态漂移、LVD 边际侵蚀和局部干扰尖峰？
2. 所提出的损失是否能在保持目标任务学习效果的同时，减少安全核心越界、Token 胞腔翻转、干扰各向异性和总体激活漂移？
3. 该方法相对数据混合、蒸馏约束、梯度正交、PEFT、参数剪枝、模型合并和可解释性方法，在效果、算力、显存和稳定性上处于什么位置？

本文档定义工程阶段、模块边界、实验契约、验收条件以及开始开发前需要补充的参考代码和环境信息。

---

## 2. 对当前四阶段规划的审查结论

### 2.1 总体结论

当前四阶段顺序没有致命问题，可以作为项目主线：

1. 训练模板；
2. 数据与基准结果；
3. LVD 与新损失；
4. 对比实验与分析。

项目不设置额外预备阶段。已经确认的工程决策作为第一步的固定基线；模型结构、离线缓存、LVD 代数、分布式和 lm-eval 检查全部并入第一步的实现与验收。

### 2.2 必须调整的事项

#### A. 不显式构造完整 LVD

真实模型的隐藏维度通常为数千维，词表规模可达数万至十余万。显式枚举并构造全部 Token 胞腔、顶点和邻接关系在计算上不可行，也不是损失计算所必需的。

工程实现应利用 Unembedding 的代数等价性：记录阶段分块计算参考 Token 到词表中每个其他 Token 的边际，训练阶段只为当次随机采样的 K 个竞争 Token gather 权重行和边际。不得构造 $V\times V$ Token 对或完整胞腔。本文中的“平面 LVD”指边界由平直仿射超平面构成，并不表示将隐藏状态降到二维平面后再计算损失。二维投影只用于可视化，不能用于训练损失。

#### B. 在第一阶段预留双数据流和自定义损失接口

新方法的每个训练步至少涉及：

- 目标任务批次，用于计算 SFT 损失；
- 通用能力锚定批次，用于计算隐藏状态漂移和 LVD 损失；
- 冻结参考模型的输出，或由参考模型预计算的缓存。

因此训练模板从第一阶段开始就必须支持双 DataLoader、损失插件、最终隐藏状态提取和额外指标回调。不能在第三阶段再整体改写训练循环。

#### C. 评测与结果记录底座不能全部推迟到第四阶段

第四阶段负责完整实验矩阵、统计分析和论文图表，但以下能力必须从第一阶段开始存在：

- 配置快照；
- Git commit、模型 revision、数据版本和随机种子记录；
- 训练/验证指标的结构化落盘；
- checkpoint 与断点续训；
- 基础通用能力和目标任务评测入口。

否则第二阶段得到的 benchmark 与后续方法结果可能不可比。

#### D. 必须明确冻结 Unembedding 的实验协议

方法推导使用固定的 Unembedding 参数 $W_0,b_0$ 作为 LVD 几何坐标系。主方法默认采用以下协议：

- 保存完整冻结参考模型 $M_0$；
- 待训练模型记为 $M_\theta$；
- 主方法训练时冻结待训练模型的 `lm_head`/Unembedding，使其保持为 $W_0,b_0$；
- Transformer 主体进行全量更新或 LoRA 更新；
- 若模型输入 Embedding 与 `lm_head` 权重绑定，主方法必须在初始化时显式解耦：复制基座输出权重形成独立 `lm_head`，将模型配置改为非绑定，验证 step 0 的 Logit 不变，再冻结复制后的输出头并允许输入 Embedding 按主体协议训练；
- 解耦后的模型必须通过保存、加载和 checkpoint 恢复测试。如果某模型适配器无法可靠解耦，则第一步的模型适配验收直接失败，不得静默冻结输入 Embedding；“同时冻结输入 Embedding 与输出头”只能作为单独命名的消融设置。

实验框架仍需支持真正的全参数微调基线，包括可训练 `lm_head`。但该基线与“冻结 LVD 的主方法”必须使用不同、准确的名称。允许增加“训练 `lm_head`、但损失仍使用冻结 $W_0,b_0$”的消融实验，不能将其默认解释为严格保护当前模型的实时决策胞腔。

#### E. 随机 K 必须能够覆盖整个词表

按基座或当前 Logit 选择 Top-K 会持续偏向常见高概率 Token，使大量低频 Token 永远没有机会参与几何约束。主实现固定采用以下协议：

1. `record_anchor_d0` 为每个保留位置记录参考 Token 到全部 $V-1$ 个竞争 Token 的 $d^{(0)}$，按 token ID 索引；
2. 每次训练访问该位置时，从参考 Token 之外的完整词表中均匀、无放回随机采样 K 个竞争 Token，采样不读取基座或当前 Logit 排名；
3. K 只属于训练配置，不属于缓存身份，因此同一份全词表 d0 缓存可用于不同 K 的实验。

若同一位置被访问 $T$ 次，任一给定竞争 Token 从未被抽中的概率为 $(1-K/(V-1))^T$。正式实验必须记录随机种子、累计唯一竞争 Token 数及理论期望覆盖率，并用不同 K 做消融。

有限次随机采样仍不等价于逐步检查完整词表。全词表校验固定在版本化诊断子集上执行：每次正式评测 checkpoint 使用完整当前 Logit 找到真实获胜 Token，并报告翻转位置数、完整词表 crossing rate、随机样本是否命中真实翻转 Token及置信区间。只有完整词表诊断支持时，才能对“防止胞腔越界”作强结论。

#### F. 两项理论表述需要作为待验证假设，而不是工程先验

- 有限权重的平滑指数惩罚可以强烈抑制越界，但不等价于数学上的硬约束或绝对不越界。论文中应报告实际 crossing rate，而不是仅凭函数形式宣称绝对保证。
- 最小化协方差 Frobenius 范数会同时降低总压强和最大特征值。在总能量固定时，各向同性是最优解；总能量不固定时，优化器也可能主要通过缩小全部压强来降低损失。因此必须同时报告总量指标和归一化各向异性指标，验证是否真的发生了“均分”。

#### G. “统一协方差”公式在统计学上是未中心化二阶矩

原方法使用 $\Sigma=N^{-1}P^\top P$，没有减去样本均值。本文沿用 `Unified-Cov` 作为方法名称，但代码、日志和论文公式必须将其准确标注为 `uncentered_second_moment`。中心化协方差应作为消融实现，不能在未说明的情况下替换主方法公式。

以上问题不会阻止项目开发，但会影响实验命名、代码接口和最终论文结论，必须在实现前固定协议。

### 2.3 三份参考代码的适用边界与取舍

已审查以下本地快照：

- `Code_for_reference/SMART-main`：data mixture 与 Accelerate SFT；
- `Code_for_reference/Future_localization-master`：组件、边和神经元级归因；
- `Code_for_reference/Geometric-Lens-main`：Unembedding 读出与二维决策边界可视化。

三份代码均提供了有价值的局部模式。补齐后的 Future Localization 微调框架可以作为第一步的结构与行为蓝本，但不能原样作为最终工程底座；主项目对其训练主干做最小化重构，并用 SMART 与 Geometric Lens 做数据和几何对照。

| 参考工程 | 保留的启发与可对照接口 | 明确不纳入核心实现 |
|---|---|---|
| SMART | Accelerate 自定义循环、response-only label mask、任务选择到实例预算的两阶段数据血缘、训练资源日志 | 固定 8 卡配置、旧式 Hub Git 工作流、当前快照主训练路径中未接入模型构造的量化配置、脆弱 resume 逻辑、pickle ordering、首版重新运行完整 FAISS/submodlib 数据选择 |
| Future Localization | 以 `src/exec/finetuning_model.py`、`src/model/finetuning.py`、`src/finetuning`、`src/dataset` 为蓝本的配置/训练/评测主干；Target FT、Anchor CE/KL、standard PEFT LoRA、结构化 batch 与 checkpoint/eval 编排；另保留 Hook 归因和指定版本 lm-eval 的设计参考 | mask/GenerateMask/WANDA/SNIP、梯度 mask、head/projection LoRA、按归因分数分配 rank、两阶段交替训练、Sophia、probing、unlearning、任务特化 Edge-Pruning、全参数 dense mask、默认图导出和主训练内 HVP；裁剪后的 lm-eval 目录不能未经恢复和验证直接加入 `PYTHONPATH` |
| Geometric Lens | 最终归一化层到 Unembedding 的读出路径、Logit/Top-K 局部对照、二维决策区域和轨迹的展示思路 | 将 notebook 作为正式入口、pickle/部分隐藏状态默认降精度导出、手写全局模型分支、AE 训练代码直接复用、Patchscopes/Jacobian Lens/逐层 ablation 进入 MVP |

审查得到的关键结论如下：

1. Geometric Lens 参考代码没有实现本项目所需的 $c_j=W_j/2$、$\nu_j=b_j+\|W_j\|_2^2/4$、局部超平面距离、margin erosion 或 Unified-Cov Loss。第三阶段必须依据公式独立实现，不能把该仓库视为 LVD loss 的现成模板。
2. SMART 的首要价值是数据选择工件的依赖链，而不是其训练脚本。项目首版直接消费已发布 SMART 数据；只有需要重建 mixture 时才引入任务嵌入、预算分配和实例排序依赖。
3. Future Localization 新补齐的 `src` 微调框架作为第一步的工程蓝本，但只抽取配置分层、目标/通用数据 schema、损失分派、standard LoRA、保存和评测主干；不直接复制其巨石 `Finetuning` 类和手写单卡训练循环。Future Localization 的归因部分仍只作为第四步离线诊断参考，不进入每个训练 step，也不得改变优化器参数组。
4. 所有正式功能必须有可测试的 Python/CLI 入口。Notebook 只用于探索和展示，不能成为产生论文数字的唯一入口。
5. 三个顶层仓库均为 MIT 许可，但 Future Localization 和 Geometric Lens 内含带 Apache-2.0 文件头的第三方代码。项目优先独立实现；若复制实质代码，必须按具体来源保留版权和许可证声明，并在上游实际提供 NOTICE 文件时按许可证要求传递相应 NOTICE 内容。
6. 当前参考目录是本地代码快照，不应假定其分支或 commit。正式引用前需要记录上游 URL、版本/commit；若无法获得，则生成文件清单和内容哈希作为 snapshot revision。
7. `Future_localization-master/lm-evaluation-harness` 被指定为本项目通用能力评测工具的来源基线。其环境文件固定 `lm-eval==0.4.9`，并给出了初始通用任务组；当前目录是裁剪快照，缺少包元数据、CLI、`lm_eval/utils.py` 及计划任务目录，因此必须从 EleutherAI 官方 0.4.9 对应 revision 恢复完整工作树、审计本地差异并通过兼容性测试后使用，不能把不完整快照的导入失败误判为评测框架不可用。
8. 新框架的可复用主链为 `CLI/config -> model factory -> target/anchor dataset -> SFTConflictTrainer -> checkpoint/evaluation`。其中 `SFTConflictTrainer` 借鉴 `BaseFinetuningTrainer` 和 `TargetFT*` 的损失分派，但使用标准 `Trainer.train(resume_from_checkpoint=...)` 执行训练，由 Trainer 底层 Accelerate 负责 DDP/FSDP、梯度累积、scheduler 和状态恢复。

### 2.4 首版最小范围与明确排除项

首版以 Future Localization 微调框架为蓝本，只实现证明核心假设所需的最小闭环：decoder-only Causal LM、Full FT/冻结头主体微调/standard LoRA、目标任务 SFT、SMART 锚定数据消费、全词表 d0 参考缓存、随机 K LVD、Unified-Cov Loss、任务评测和高维几何诊断。

以下功能暂不进入 MVP：

- 重新运行完整 SMART task/instance selection；
- 4-bit/8-bit、QLoRA、模型合并、参数剪枝和在线可解释性定位；
- Patchscopes、Jacobian Lens、逐层 ablation 和逻辑电路图导出；
- 全参数 dense mask、HVP/future localization 和每步组件/神经元归因；
- 用 Autoencoder 生成训练损失或以二维投影指标替代高维 LVD 指标；
- 将第三方评测源码复制进 `src/sft_conflict`、自动 push checkpoint 到 Hub、Web UI 和大而全的 Docker 镜像。恢复后的 lm-evaluation-harness 作为隔离的工具工作树或锁版本依赖运行，不成为训练包内部实现。

这些能力只在主方法闭环稳定、存在明确实验问题且能给出独立验收标准时引入。

---

## 3. 统一术语与实验对象

### 3.1 数据角色

- $D_T$：目标任务训练集，用于 SFT 主损失。
- $D_P$：通用能力锚定集（Pervasiveness/Anchor Dataset），用于参考激活和 LVD 保护损失。
- $E_T$：目标任务验证/测试集。
- $E_P$：通用能力评测集，只用于评测，不参与训练或超参数选择。
- $D_P^{cal}$：可选校准子集，用于选择 $\lambda,\gamma_0,K$ 等方法超参数，必须与 $E_P$ 分离。

四类数据必须具有稳定样本 ID、来源字段和版本信息，并执行去重与泄漏检查。

### 3.2 模型对象

- $M_0$：冻结的基座/参考模型，参数为 $\theta_0$。
- $M_\theta$：正在微调的模型。
- $h_{0,n}$：参考模型在第 $n$ 个锚定位置进入 Unembedding 前的最终隐藏状态。
- $h_{\theta,n}$：当前模型在同一位置的最终隐藏状态。
- $\Delta h_n=h_{\theta,n}-h_{0,n}$：激活漂移。
- $W_0,b_0$：冻结的 Unembedding 参数，也是 LVD 的固定几何坐标系。

需要明确“第 $n$ 个锚定位置”的粒度。框架必须支持：

- `last_non_padding`：每个序列最后一个有效位置；
- `all_selected_tokens`：所有满足掩码条件的位置；
- `sample_k_positions`：按固定随机种子从每个序列选择若干位置，建议用于首版以控制显存和计算量。

### 3.3 核心结果指标

- 目标任务效果：任务原生指标，如 Accuracy、F1、Exact Match、pass@k、win rate。
- 通用能力保留：微调前后绝对变化、相对变化和平均保留率。
- LVD 指标：原始边际、残留边际、安全核心违反率、胞腔 crossing rate、随机竞争 Token 的累计词表覆盖率和完整词表 crossing 诊断。
- 干扰指标：压强向量模长、总二阶矩、最大特征值、谱有效秩、归一化各向异性。
- 漂移指标：$\|\Delta h\|_2$ 的均值、分位数和层/Token 分布。
- 工程指标：每步耗时、吞吐量、峰值显存、checkpoint 大小和额外存储。

---

## 4. 第一步：以 Future Localization 为蓝本重构微调框架

项目直接从第一步开始。第一步同时完成蓝本裁剪、训练框架重构和必要冒烟检查；只有第一步验收通过后，才进入正式数据与 benchmark 阶段。

### 4.1 蓝本主链与迁移原则

参考框架的真实主链为：

```text
src/exec/finetuning_model.py
	-> src/model/finetuning.py::Finetuning
	-> src/dataset.get_finetuning_dataset
	-> src/finetuning.get_finetuning_method
	-> BaseFinetuningTrainer / TargetFT*
	-> train / save / eval / lm-eval
```

本项目保留这条主链的职责分层，但不直接复制巨石类。新主链为：

```text
CLI + YAML/dataclass config
	-> model/tokenizer factory
	-> target + anchor data module
	-> SFTConflictTrainer (HF Trainer subclass)
	-> loss plugins (Target CE / Anchor CE / Anchor KL / LVD)
	-> checkpoint + evaluation orchestrator
```

`SFTConflictTrainer` 以 Future Localization 的 `BaseFinetuningTrainer`、`TargetFT`、`TargetFT_PervasivenessFT` 和 `TargetFT_PervasivenessKL` 为设计蓝本；训练执行统一调用标准 `Trainer.train(resume_from_checkpoint=...)`，由 Hugging Face Trainer 底层 Accelerate 负责单卡、DDP/FSDP、梯度累积、scheduler、混合精度和训练状态恢复，不沿用参考代码中 `device_map={"": 0}` 加手写 DataLoader/optimizer 循环的实现。

需要特别区分“职责蓝本”和“现有运行机制”：Future Localization 虽构造了 `Trainer` 子类，但当前实际训练由 `src/model/finetuning.py::_run_finetuning_training` 的手写 DataLoader/optimizer 循环执行，并未调用 `Trainer.train()`；新项目只继承损失分派和编排思想，不继承该训练机制。

### 4.2 保留、重写与删除范围

#### 4.2.1 保留并收敛

- `overall / finetuning / dataset / evaluation / logger` 的配置分区思想，改写为 YAML + dataclass；
- `target` 与 `pervasiveness` 的结构化 batch 语义，统一改名为 `target` 与 `anchor`；
- `TargetFT`、`TargetFT+PervasivenessFT`、`TargetFT+PervasivenessKL` 的损失分派，改名为 `target_only`、`target_plus_anchor_ce`、`target_plus_anchor_kl`；
- `training_losses.py` 中标签清洗、Causal LM shift 和 KL 的纯函数思路，经独立单元测试后迁移；
- PEFT 的 standard LoRA 路径，只保留固定 rank/alpha/dropout/target modules；
- checkpoint、epoch eval、最终评测和 lm-eval 编排的职责边界；
- JSON 结果落盘思路，升级为 JSONL/Parquet + TensorBoard。

#### 4.2.2 必须重写

- 将 `src/model/finetuning.py::Finetuning` 拆分为模型工厂、数据模块、Trainer、loss plugin、checkpoint 和 evaluator；
- 将 Fastargs 和运行时 `sys.path.append("src")` 改为可安装 Python package、YAML + dataclass 和正式 console entrypoint；
- 将固定 BF16、固定 `cuda:0` 和 `device_map` 改为 Trainer/Accelerate 的设备与精度配置；训练时禁止用 `device_map="auto"` 做模型并行替代 DDP/FSDP；
- 将 `resume_path` 的“只加载后评测”语义拆成 `resume_from_checkpoint`（继续训练）和 `eval_checkpoint`（只评测）；
- 将长度取最小值、按 modulo 配对的 `FinetuningDataset` 改为显式 target/anchor 双流采样器，独立 batch size、比例、epoch 长度和分布式 sampler；
- 将任务特化的最后一个 Token 标签改为通用 response/assistant mask，同时保留 next-token 任务适配接口；
- 将 `print` 型训练日志改为 Trainer callback 和结构化 run metadata；
- 在模型工厂中新增离线 cache、Qwen2.5 适配、Unembedding 冻结、tied-weight 解耦和实际 `lm_head` 输入捕获；
- 在 loss plugin 接口中预留第三步 LVD Loss 所需的 Anchor 元数据和隐藏状态，不在第一步实现完整 Unified-Cov。

#### 4.2.3 从主框架删除或延后

- 所有 `mask_path`、GenerateMask、WANDA、SNIP/SNIP-joint、`mask_gradient`、mask cache 和按 mask 切换任务；
- `projection_matrix`/`head` LoRA、component-wise LoRA、rank/alpha pattern、EAP score 驱动 rank 和多任务 LoRA 切换；
- `two_stage_alternating`、stage/task 轮换和 conflict/all-component 路径；
- Sophia、probing、L1 变体、unlearning、pruner 和恢复训练以外的 recovery 逻辑；
- MIA、PII、toxicity、WMDP、ToFU、Harry Potter、SafePKU、IOI、induction 等与首版任务无关的数据和指标；
- 默认 Mistral RoPE patch、Blackwell 特例、CUDA DSA/`CUDA_LAUNCH_BLOCKING=1`；只有目标模型确有需要时才以独立适配器启用；
- 自动 merge LoRA 后覆盖训练中 `self.model` 的行为；Adapter 保存与 merged 导出必须是两个独立产物。

被删除参数不得继续出现在新 CLI、配置 schema、run metadata 或默认配置中；不能保留“无效果兼容参数”。

### 4.3 第一步前置检查与最小闭环

第一步实现过程中直接完成以下检查，不另设阶段：

1. 使用 Toy LM 验证最大线性 Logit、最小 Laguerre 距离和成对超平面符号一致；
2. 使用 `Qwen/Qwen2.5-0.5B` 检查 final norm、实际 Unembedding 输入、bias、tied weights、冻结/解耦和 round-trip Logit；
3. 分别用 SST-2 与 GSM8K 固定 256 条完成 target-only Full FT 和 standard LoRA 20 至 100 step 冒烟；
4. 使用 256 条 Anchor 完成 `target_plus_anchor_ce`、`target_plus_anchor_kl` 和空 LVD plugin 双流冒烟；
5. 验证 checkpoint 保存、optimizer/scheduler/RNG/global step 恢复后继续训练，而不仅是加载后评测；
6. 验证 1 卡无包装路径和 2 卡 DDP；FSDP 在 3B Full FT 前完成最小验证；
7. 验证离线 cache 缺失时 fail-fast，缓存完整时全程无 Hub 请求；
8. 恢复 lm-eval 0.4.9 并完成 HellaSwag/GSM8K 限制集评测；
9. 为三份参考快照生成 hash 和许可证清单，参考目录不加入运行时 `PYTHONPATH`；
10. 输出机器可读的模型结构、最终配置、可训练参数、数据指纹和依赖版本。

### 4.4 已确认的项目工程基线

以下决策于 2026-08-22 固定。后续若修改，必须通过配置显式覆盖，并在 run metadata 中保存最终解析值；不得只修改代码默认值。

#### 4.4.1 环境、硬件与存储

| 项目 | 已确认方案 |
|---|---|
| 主训练框架 | 以 Future Localization 为蓝本的最小 Hugging Face `Trainer` 子类；Accelerate 作为 Trainer 的分布式后端，不维护第二套手写训练循环 |
| 环境隔离 | 训练环境与 `lm-eval==0.4.9` 评测环境分离并分别锁版本 |
| 训练精度 | 命令行/配置可选 `bf16`、`fp16`、`fp32`，默认 `bf16`；LVD 归一化、距离、二阶矩和指数运算强制 FP32 |
| GPU | 单机最多 4×NVIDIA A100-SXM4-40GB，Driver 535.129.03，驱动报告 CUDA 12.2，MIG 关闭 |
| 并行策略 | 0.5B/LoRA 默认 DDP；3B/7B Full FT 默认 FSDP Full Shard；支持 1 至 4 卡 |
| 配置系统 | YAML + dataclass；首版不引入 Hydra |
| Artifact 根目录 | `/data1/chenhang/SFT_Conflict/artifacts`，位于 Git 仓库之外 |
| 文件系统容量 | `/data1` 为 14 TB ext4；检查时已用 2.9 TB、可用约 11 TB、inode 使用约 1%。该数值只代表检查时快照，实际运行以容量闸门为准 |
| 容量告警 | 每次正式运行前估算产物；若可用空间低于 2 TB 或低于本次预计新增量的 2 倍，停止启动并反馈 |
| Python/CUDA 环境现状 | 系统仅有 Python 3.10.12，尚无 PyTorch；实现阶段创建独立环境，并验证 PyTorch CUDA 与 BF16 支持 |

Artifact 默认布局：

```text
/data1/chenhang/SFT_Conflict/artifacts/
├── hf_cache/
│   ├── hub/
│   └── datasets/
├── reference_cache/
├── checkpoints/
├── evaluation_cache/
├── results/
└── tmp/
```

集中设置：

```text
HF_HOME=/data1/chenhang/SFT_Conflict/artifacts/hf_cache
HF_HUB_CACHE=/data1/chenhang/SFT_Conflict/artifacts/hf_cache/hub
HF_DATASETS_CACHE=/data1/chenhang/SFT_Conflict/artifacts/hf_cache/datasets
```

按当前网络策略，训练与评测默认不访问 Hugging Face；检查时 cache 为空、现有 `127.0.0.1:1081` 代理不可用。准备阶段先探测届时可用的下载通道；若仍不可直连，则通过镜像或从其他机器迁入完整 snapshot。缓存 manifest 必须记录 repo ID、repo type、commit、文件清单和哈希。训练/评测阶段统一设置：

```text
HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

并使用 `local_files_only=True`。cache 缺失或 revision 不匹配时立即失败，不在训练中自动联网补文件。

#### 4.4.2 模型、任务与数据

| 项目 | 已确认方案 |
|---|---|
| 冒烟模型 | `Qwen/Qwen2.5-0.5B` Base |
| 首轮正式模型 | `Qwen/Qwen2.5-3B` Base；闭环稳定后扩展 `Qwen/Qwen2.5-7B` |
| 模型范围 | 首版完整支持 Qwen2.5；Base 为主实验，Instruct 为扩展实验；所有 revision 固定 commit |
| 冒烟目标任务 | 同时支持 SST-2 和 GSM8K，各使用固定 256 条样本 |
| 首个正式目标任务 | GSM8K；pilot 使用 2,000 条，正式实验使用完整训练集 |
| 最大序列长度 | 冒烟 512；GSM8K pilot/正式实验 1024 |
| SFT 标签 | 仅 response/assistant Token 计算损失 |
| 双流比例 | 每个目标任务批次对应一个 Anchor 批次，即 1:1 |
| Anchor 数据 | `kowndinya23/flan2022-4096-1840-tasks-logdet-400000-instances-logdet`，下载时固定 dataset commit |
| Anchor 规模 | 冒烟 256、pilot 5,000、首轮正式 20,000 条分层样本 |
| Anchor 位置 | 每条样本按固定 seed 采样 4 个 response Token，排除 PAD/EOS 和无监督位置 |
| 几何参考 Token | 使用基座模型 argmax；主实验仅保护基座 next-token 预测正确的位置，全部位置作为消融 |
| Anchor 采样 | task-balanced 为主；mixture-proportional 作为消融 |
| 参考计算 | 冒烟使用在线参考验证；正式实验使用预计算缓存 |
| 缓存精度 | 初始 FP32；通过预注册误差预算后允许 BF16，FP16 仅作额外候选 |
| 元数据 | 保留 task/source/template/stable sample ID，不删除来源字段 |

#### 4.4.3 Unembedding、LVD 与训练默认值

| 项目 | 已确认方案 |
|---|---|
| Unembedding/`lm_head` 冻结 | 暴露 `--freeze-unembedding/--no-freeze-unembedding`，默认冻结；真正 Full FT baseline 显式传入 `--no-freeze-unembedding` |
| tied weights | 按训练模式处理：真正 Full FT 保留原生权重绑定；冻结头且输入 Embedding 可训练时先解耦并只冻结输出头；LoRA 默认保持原生绑定和冻结基座，只有输入 Embedding 将被训练时才解耦。所有行为配置化并记录 |
| 隐藏状态 | 捕获 final norm 后、实际进入 Unembedding 的张量 |
| 邻居策略 | 从参考 Token 之外的完整词表均匀无放回随机采样 |
| 竞争方向 K | 当前 logits Top-K 默认 512；随机 K 保留为消融；数量可在 run 的 LVD_DIRECTION 修改 |
| 安全核心 | 默认 $\gamma_0=0.2$ |
| 压强参数 | 默认 $\mu_3=1$、$\mu_4=2$、$p=1$，指数输入默认 clamp 到 20 |
| 损失权重 | 先运行 target-only pilot，再将 $\lambda$ 校准到辅助损失梯度范数约为 SFT 梯度范数的 50%；使用固定校准批次和全部共享可训练参数，聚合规则见第三步“统一协方差损失”；最终值写入配置快照 |
| 二阶矩形式 | 未中心化二阶矩为主方法；中心化协方差只作消融 |

冻结相关设置全部由命令行/配置解析，至少包括训练模式、Unembedding 冻结和 tied-weight 解耦。程序必须记录最终可训练参数清单；run 名称依据解析后的真实参数状态生成，不能仅依据用户填写的模式名称。

#### 4.4.4 预注册验收门槛

| 项目 | 已确认方案 |
|---|---|
| 随机邻居覆盖 | 报告累计唯一 Token 数及理论期望覆盖率；完整词表诊断不足时不能作全词表保护结论 |
| Crossing 改善 | 相对 `target_only` 至少降低 20%；若 baseline crossing 为 0，则改用预注册的 safety-core violation/margin erosion 指标 |
| 目标任务退化上限 | 不超过 1 个绝对百分点或基线指标的 2% 相对值，取较宽者 |
| 正式随机种子 | 3 个；冒烟阶段 1 个 |
| 统计方法 | per-sample paired bootstrap 95% CI，并同时报告效果量 |
| 全词表诊断规模 | 每个正式 checkpoint 固定检查至少 1,000 个 Anchor 位置 |

#### 4.4.5 评测、日志与产物

| 项目 | 已确认方案 |
|---|---|
| lm-eval 后端 | 从本地裁剪快照恢复官方 `lm-eval==0.4.9` 完整工作树，并通过对齐测试 |
| lm-eval 冒烟任务 | HellaSwag + GSM8K 限制集，分别覆盖 loglikelihood 与 generation |
| 第一版正式任务 | `boolq,rte,hellaswag,winogrande,arc_challenge,arc_easy,openbookqa,piqa,truthfulqa` |
| few-shot | 冒烟全部 0-shot；正式评测使用固定 0.4.9 task YAML 默认值（`num_fewshot=None`），并保存 YAML hash |
| 模板 | Base 模型不应用 chat template；Instruct 使用模型官方 chat template，分开报告 |
| 汇总指标 | 每任务单独报告；宏平均相对保留率只作为辅助总指标 |
| 评测频率 | 小诊断集每 epoch；完整任务组只评测 Base 和最终 checkpoint |
| 日志 | 本地 JSONL/Parquet + TensorBoard；W&B 默认关闭 |
| Checkpoint | 冒烟只保存最终；正式实验每 epoch 保存，保留最近 2 个和最终/最优产物 |
| 确定性 | Toy 测试严格确定；GPU 正式训练 best effort，并完整记录版本与非确定算子 |
| 参考快照 | 能获得上游 URL/commit 时记录；否则使用当前内容的 snapshot hash |
| lm-eval 恢复时间 | 第一步完成，避免到第二步才发现兼容问题 |

---

### 4.5 第一步详细功能与验收

#### 4.5.1 目标

建立一个可扩展的统一训练入口。同一个 `SFTConflictTrainer` 通过配置切换普通全参数微调、冻结头的主体微调和 standard LoRA，并通过 loss plugin 切换 target-only、Anchor CE/KL 及后续 LVD Loss。

训练框架以 Future Localization 的配置分层、`BaseFinetuningTrainer` 和 `TargetFT*` 为蓝本，重构为最小 Hugging Face `Trainer` 子类。Trainer 底层 Accelerate 提供单卡/DDP/FSDP 与混合精度；不保留参考代码的手写训练循环，也不另建一套并行训练实现。TRL 不作为核心依赖。

#### 4.5.2 模型与 Tokenizer 功能

1. 按 model ID/revision 加载模型和 Tokenizer。
2. 支持 Base 与 Instruct 模型：Base 的对应策略是使用任务定义的纯文本 prompt、不应用 chat template；Instruct 显式应用模型官方 chat template。两者分开配置和报告。
3. 正确设置 `pad_token`、`eos_token`、padding side 和截断策略。
4. 提供 `--mixed-precision {bf16,fp16,fp32}`（或等价 YAML 字段），默认 `bf16`；几何损失中的关键归一化、距离、二阶矩和指数运算强制 FP32。
5. 输出模型结构审计结果：参数总量、可训练参数量、`lm_head` 状态、权重绑定状态和 LoRA 注入模块。
6. 支持以下训练模式：
	- `full_ft`：保留模型原生 tied-weight 关系，所有参数可训练，包括 `lm_head`，仅作为真正全参数基线；
	- `body_ft_frozen_head`：Transformer 主体可训练，Unembedding 冻结；
	- `lora`：仅使用 standard PEFT LoRA，默认冻结原模型和 `lm_head`，只训练指定 Adapter；
	- 后续可扩展 QLoRA，但不作为首版正确性实现。
7. 通过 `get_output_embeddings()`、实际模块 Hook 和适配器注册表定位 Unembedding，不依赖 model name 前缀或全局变量。
8. 在 step 0 捕获 `lm_head` 的实际输入 $h$，验证 $F.linear(h,W,b)$ 与模型的原始线性 Logit 一致，并验证其 argmax 与最终输出 Logit 一致。若存在 soft-cap，仅当它对所有类别使用相同严格单调函数、因而保持 argmax 时才允许继续；几何边际必须使用变换前线性 Logit。
9. Unembedding 冻结必须使用 BooleanOptionalAction 或等价双向开关，默认 `freeze_unembedding=true`；`full_ft` 配置显式覆盖为 false，其他模式不得依赖隐含分支改变该值。
10. tied-weight 策略由训练模式解析：`full_ft` 保持原生绑定；`body_ft_frozen_head` 在输入 Embedding 可训练时解耦并冻结复制后的输出头；默认 LoRA 不训练输入 Embedding，因此保持原生绑定并冻结基座。若 LoRA 配置要求训练输入 Embedding，必须先解耦。保存、加载及 merge 后都要复核绑定状态和 step-0/round-trip Logit。

#### 4.5.3 Standard LoRA 功能

1. 通过配置定义 target modules，不在代码中写死模型层名称。
2. 支持 rank、alpha、dropout、bias 策略和 modules-to-save。
3. 输出实际命中的模块列表，若 target module 一个都未命中必须立即失败。
4. 支持 Adapter 单独保存、重新加载和可选 merge。
5. 验证 merge 前后同一输入的 Logit 在容差范围内一致。
6. 在 optimizer 创建后断言所有且仅有预期 Adapter 参数位于 param groups 中；保存/加载后重复断言。
7. 首版 schema 不包含 `lora_mode`、rank pattern、alpha pattern、component score、head rank 或 task-specific LoRA info 路径；这些解释性变体从代码和配置中删除，而非设置为隐藏开关。

#### 4.5.4 Trainer 与训练生命周期

1. 通过 `TrainingArguments`/等价 typed config 支持 gradient accumulation、gradient clipping、学习率调度、warmup 和 weight decay。
2. 通过 Trainer/Accelerate 支持 1 至 4 张 GPU；使用 1 卡时不启用 DDP/FSDP 包装；0.5B/LoRA 在 2 至 4 卡时默认 DDP，3B/7B Full FT 在 2 至 4 卡时默认 FSDP Full Shard。模型加载阶段不传训练用 `device_map`。
3. 调用标准 `Trainer.train(resume_from_checkpoint=...)`，checkpoint 必须包含模型/Adapter、optimizer、scheduler、scaler、Trainer state、global step、epoch、双流 sampler/RNG 状态和已解析配置；`eval_checkpoint` 是独立只评测参数。
	对 flat-parameter FSDP 的块量化 optimizer，允许使用版本化的 rank-local optimizer shard，
	但必须原子验证所有 rank 文件、绑定 optimizer 类型与 world size，并在配置或 rank 数变化时
	fail-fast；不得因上游全量 gather 不兼容而静默省略 optimizer state。
4. 支持目标批次与锚定批次的双流迭代：
	- 独立 batch size；
	- 可配置采样比例；
	- 某个 DataLoader 先耗尽时可循环或重采样；
	- 梯度累积期间保持 loss scale 一致。
5. 提供 `Regularizer`/`AuxiliaryLoss` 插件接口，输入至少包括模型、当前隐藏状态、批次元数据和全局 step，输出标量损失与结构化诊断指标。
6. `lambda=0` 时，带插件训练路径必须与普通 SFT 路径数值对齐。
7. 训练期间不能默认返回所有层 hidden states，以免造成额外显存占用；只捕获进入 Unembedding 前的最终状态。
8. Hook 必须由上下文管理器注册和移除，支持模块输出为 Tensor/tuple/model output，并只截取被选中的 Token 位置；异常退出后 Hook 数量必须恢复为零。
9. 每个 optimizer step 后校验 `requires_grad` 白名单与 optimizer param groups 未发生漂移。任何诊断插件不得像 Future Localization 的 future 路径一样永久打开冻结参数梯度；需要临时梯度时必须保存并恢复模型模式、参数状态和现有梯度。
10. 所有可选能力必须有生效性断言：量化检查实际模块类型和权重 dtype，gradient checkpointing 检查模型状态，Flash Attention 检查实际 attention backend，PEFT 检查 Adapter 注入与非零梯度。配置开关存在但未进入模型构造时必须失败。
11. `compute_loss` 只负责组合已注册 loss term，并返回各分项诊断；不得加载 mask、动态切换 LoRA 模块或在 batch 间修改 `requires_grad`。
12. target 与 Anchor 数据流由自定义 `get_train_dataloader`/batch scheduler 组合，不能复用参考框架以 `min(task_lengths)` 定义 epoch、再用 modulo 重复短数据集的隐式语义。

双流 sampler state 至少序列化：target/Anchor 各自 dataset fingerprint、epoch、shuffle seed、当前 cursor、已消费样本数、循环次数、当前 permutation/generator state、world size、rank、各 rank shard 起止/offset、双流采样比例和 gradient-accumulation micro-step。恢复时任一数据 fingerprint、world size 或采样配置改变均默认 fail-fast；如允许弹性恢复，必须作为单独模式并重新建立数据位置，不能宣称与不中断运行数值等价。

#### 4.5.5 配置与日志

每次运行必须保存：

- 完整解析后配置；
- Git commit 和 dirty 状态；
- Python/CUDA/PyTorch/Transformers/PEFT 版本；
- model ID、model revision 和 tokenizer revision；
- 数据集名称、revision、样本清单哈希；
- 随机种子、GPU 型号和数量；
- 可训练参数列表摘要；
- 训练、验证和资源指标的 JSONL 或 Parquet；
- 可选的 W&B/TensorBoard 同步，但本地结构化结果是强制项。

默认启用 TensorBoard、关闭 W&B。所有路径解析到 `/data1/chenhang/SFT_Conflict/artifacts` 下的对应子目录，运行前执行磁盘容量预检。

新配置中禁止出现 mask、WANDA/SNIP、probing、Sophia、unlearning、head/projection LoRA 和 two-stage alternating 字段。加载旧参考配置时必须由显式迁移工具生成新配置并报告被丢弃字段，主训练 CLI 不直接兼容旧参数。

#### 4.5.6 测试与验收

1. 单批次前向与反向传播成功，无 NaN/Inf。
2. 在极小数据集上过拟合，训练损失明显下降。
3. 固定种子重复短训练，关键结果在设定容差内一致。
4. checkpoint 恢复后的 step、optimizer、scheduler 和随机状态正确。
5. 三种训练模式的可训练参数集合符合预期。
6. LoRA 保存、加载和 merge 测试通过。
7. 双数据流与空 Auxiliary Loss 完成冒烟测试。
8. 单卡和多卡各完成固定 20-step 回归，样本顺序一致时 loss、更新步数和关键诊断在预设容差内一致。
9. 模型冻结白名单、Hook 清理和可选功能生效性测试通过。
10. `resume_from_checkpoint` 继续训练后的 global step、optimizer、scheduler、数据采样位置和固定输入 Logit 与不中断运行在容差内一致；仅加载 checkpoint 做评测不算断点续训测试。
11. 新训练入口在 import graph、CLI help、解析后配置和运行日志中均不出现已删除的解释性 mask/LoRA 变体模块；解析后配置若出现 `mask_path`、`mask_ratio`、`mask_score_type`、`lora_mode`、`lora_info_dir`、`rank_pattern`、`component_scores`、`two_stage_alternating`、`sophia` 或 `probing` 等旧键，必须失败并提示使用配置迁移工具。
12. 参考框架与新框架在同一 Toy/Qwen 0.5B target-only batch 上的未更新初始 loss 对齐；之后以新框架行为和测试作为唯一规范。

#### 4.5.7 阶段产物

- 统一训练 CLI；
- 模型和 Tokenizer 适配层；
- Full FT/LoRA 配置模板；
- checkpoint 与日志模块；
- 单元测试和最小训练脚本；
- 从旧 Future Localization 配置到新最小配置的单向迁移/差异报告；
- 一份短训练复现报告。

---

## 5. 第二步：数据接口、评测接口与基准结果

### 5.1 目标

建立目标任务、锚定数据和纯评测数据的统一注册接口，完成未经保护的 Full FT/LoRA benchmark，并为 LVD 损失生成稳定、可追踪的参考激活数据。

### 5.2 统一数据协议

原始样本至少转换为以下逻辑字段：

```text
sample_id
dataset_name
dataset_revision
task_name
split
messages 或 prompt/response
loss_mask
source_metadata
```

Tokenization 后至少提供：

```text
input_ids
attention_mask
labels
selected_anchor_positions（仅锚定数据需要）
sample_id
```

所有预处理函数必须是确定性的，并将模板版本、最大长度、截断规则和过滤规则纳入数据指纹。

标签掩码不能照搬 SMART 中“分别 tokenize prompt，再按 prompt token 数屏蔽拼接序列”的实现，因为 BPE 在 prompt/response 边界可能重新分词。优先使用 tokenizer/chat template 提供的 assistant mask；否则使用可验证的字符 offset mapping 或显式特殊分隔符构造 response-only mask。必须测试边界空格、Unicode、多轮对话、EOS、左/右截断，并对截断后没有任何监督 Token 的样本 fail-fast 或按记录规则过滤。

### 5.3 数据集注册与职责分离

1. 建立 `DatasetRegistry`，通过配置加载数据集，不在训练脚本中写数据集特例。
2. 冒烟阶段同时实现 SST-2 与 GSM8K 适配器，各固定 256 条；正式实验先以 GSM8K 形成闭环，再扩展到：
	- 数学：GSM8K；
	- NLU：GLUE 子任务；
	- 代码：CodeAlpaca/Infinity-Instruct Code，评测 HumanEval/MBPP；
	- 逻辑：LogiCoT、ScienceQA、BBH；
	- 多跳/长推理：HotPotQA 等；
	- 指令遵循：对应可稳定复现的评测集。
3. 通用锚定集使用 `kowndinya23/flan2022-4096-1840-tasks-logdet-400000-instances-logdet`；首版不要求重新运行完整 SMART 子模优化，缓存准备时固定 dataset commit。
4. 通用评测集可从 MMLU、CSQA、BBH 等开始，但不得与锚定训练样本重叠。
5. 对训练集、锚定集、校准集和测试集执行精确去重与近似去重，并生成审计报告。
6. 每个适配器显式声明所需 split。缺少 validation 时只能按配置选择“固定 seed 分层切分”“跳过训练期 eval”或“立即报错”，不得隐式读取不存在的 split。
7. 保留 `task_source`、`task_name`、`template_type` 等来源元数据。SMART 最终数据脚本为训练便利删除了这些列，本项目必须将其保存在 manifest/metadata 中，以支持任务分层采样和退化归因。

### 5.3.1 SMART 数据工件与血缘

首版将 SMART 发布数据视为外部不可变输入，只消费固定 revision，不把 `submodlib`、FAISS、句向量模型或 ordering 生成加入训练依赖。每个发布 mixture 至少生成以下 manifest：

- 上游 dataset ID、revision、split 和样本内容哈希；
- SMART 上游仓库 URL/commit 或本地 snapshot hash；
- task selection 方法、任务数、总实例预算和随机种子；
- task marginal gain 与 task budget 表；
- task/template budget 表；
- instance ordering 方法及 ordering 工件哈希；
- 最终稳定 sample ID 清单、各层级实际计数和预算守恒检查；
- 数据过滤、重命名、去重和 shuffle 规则。

如果后续重建 mixture，任务选择、预算分配、实例排序和最终 materialization 必须是四个独立可恢复步骤，使用开放 schema 而非 pickle。每步均需验证输入/输出数量、预算总和、sample ID 唯一性和 ordering 完整性；异常不得通过 bare `except` 跳过任务。SMART 参考实现中的固定 GPU 列表、`n-1` ordering 和 `zip` 分块截断行为不能继承。

由于 Unified-Cov Loss 依赖批内二阶矩，Anchor DataLoader 必须提供可配置的 task-balanced/mixture-proportional 分层采样，并记录每批任务组成。正式实验固定一种主采样协议，随机采样只作为消融，避免将批次组成变化误解释为损失函数效果。

### 5.4 锚定位置与参考缓存

参考模型计算应在 `torch.no_grad()` 和 eval 模式下进行。每个被选中的锚定位置至少需要以下信息：

- `sample_id` 和 token position；
- $h_{0,n}$；
- 参考 Token $j_n^*=\arg\max_j z_{0,j}(h_{0,n})$；
- 按 token ID 索引的全部词表基准边际 $d^{(0)}$；
- 基准参考 Logit；
- 基座预测置信度、是否与可用 gold label 一致；
- 模型、Tokenizer、数据和位置采样策略的版本信息。

锚定策略必须预先声明使用“全部参考预测”“仅基座正确位置”还是“置信度/边际过滤位置”。保护基座模型自身的预测不自动等价于保护正确能力，因此正式报告需要按基座正确性和置信度分层展示结果，且不能用 $E_P$ 的测试标签筛选训练锚点。

框架需要支持两种模式：

1. `online_reference`：训练时在线运行冻结参考模型，逻辑简单但计算和显存开销大；
2. `cached_reference`：预计算并分片保存，正式实验优先使用。

默认选择：冒烟在线参考、正式实验缓存参考；每条 Anchor 样本按固定 seed 采样 4 个 response Token，主实验仅保留基座 argmax 等于真实 next token 的位置。Anchor batch 默认 task-balanced，目标与 Anchor 批次比例固定为 1:1。

缓存建议使用 Safetensors/Parquet 等结构化分片格式，不保存 Python pickle。必须提供缓存大小估算、完整性校验、断点续生成和按 sample ID 随机读取。缓存不得提交 Git。

参考激活的存储精度必须通过误差预算决定，不能沿用 Geometric Lens 部分导出路径默认将隐藏状态转换为 FP16 的做法。先在校准子集同时生成 FP32 与候选压缩格式缓存，比较 $h_0$ 重构误差、argmax 一致率、全词表 margin 绝对/相对误差、safety-core/crossing 判定翻转率；只有全部低于预注册阈值时才允许正式使用 BF16/FP16，否则保留 FP32 或按样本保存更高精度边际。任何缓存都必须可从模型和数据 revision 重算。

### 5.5 Benchmark 训练组

第一轮至少包含：

1. `base`：未微调模型；
2. `full_ft_target_only`：目标任务真正全参数微调；
3. `body_ft_frozen_head_target_only`：冻结 Unembedding 的主体微调；
4. `lora_target_only`：LoRA 目标任务微调；
5. `target_plus_anchor_ce`：目标任务加通用锚定交叉熵；
6. `target_plus_anchor_kl`：目标任务加参考模型 KL 约束。

每个正式结果至少运行 3 个随机种子；开发阶段允许单种子冒烟测试。

### 5.6 评测接口

1. 封装统一 `Evaluator`，将生成式、选择题、分类和代码执行评测分开处理。
2. 通用能力评测的默认后端使用从 `Code_for_reference/Future_localization-master/lm-evaluation-harness` 恢复并固定到 0.4.9 的工作树，通过 `lm_eval.evaluator.simple_evaluate` 或等价 CLI 调用；不重复实现其已支持任务的 prompt、request 构造、filter、aggregation 和评分逻辑。
3. 保存聚合指标和 per-sample 结果，支持配对统计检验。
4. 记录生成参数、few-shot 数量、prompt template、最大生成长度和 stop tokens。
5. 代码执行评测必须在隔离环境中运行，并设置超时与资源限制。
6. 输出目标任务增益和通用能力损失的二维结果，而不是只报告单一平均分。
7. 第一版通用任务组采用 Future Localization 配置中的 `boolq,rte,hellaswag,winogrande,arc_challenge,arc_easy,openbookqa,piqa,truthfulqa`；扩展组再加入 MMLU、CommonsenseQA 和 BBH。启动正式实验前必须通过 `TaskManager`/`--tasks list` 确认 0.4.9 中的实际注册名，不允许因名称不存在而静默跳过任务。
8. 每个任务固定并版本化 `num_fewshot`、chat template、`fewshot_as_multiturn`、generation kwargs、filter 和 metric；Base/Instruct 模型不能在未记录的情况下共享不同 prompt 协议。
9. `--limit` 只允许冒烟测试，正式结果必须覆盖完整评测 split。正式运行启用 per-sample 输出，并根据任务支持情况执行 `check_integrity` 和 `write_out` 人工抽查。
10. Harness request cache 的 key/目录必须绑定 harness revision、任务 YAML/hash、模型 checkpoint、Tokenizer/chat template 和 few-shot 配置；任一项变化时刷新缓存。

### 5.7 测试与验收

- 每个数据适配器通过 schema、样本数、模板和 label mask 测试；
- 数据泄漏审计完成；
- Base 模型评测结果与可信公开结果或参考实现处于合理范围；
- 恢复后的 lm-evaluation-harness 可列出全部第一版任务，使用小模型完成至少一个 loglikelihood 任务和一个 generation 任务的端到端冒烟测试；
- 对选定 Base 模型运行固定小子集，结果与官方 lm-eval 0.4.9 或同 revision 独立运行在预设容差内一致；不存在的任务名、空结果、非有限指标或部分任务失败均使评测 run 失败；
- Full FT 与 LoRA benchmark 可从配置一键复现；
- 参考缓存重新加载后与在线参考输出在容差内一致；
- benchmark 结果包含至少一个明显学习到目标任务的设置，否则不进入第三阶段。

### 5.8 阶段产物

- 数据集注册与预处理模块；
- 锚定位置选择和参考缓存工具；
- 统一评测入口；
- Base、Full FT、冻结头主体微调和 LoRA 基准表；
- 数据泄漏与版本审计报告。

---

## 6. 第三步：LVD 几何框架与统一协方差损失

### 6.1 目标

在不显式构造完整高维胞腔的情况下，根据冻结 Unembedding 参数计算局部 LVD 决策超平面、边际侵蚀和容量加权压强，并将统一协方差损失接入第一阶段的训练模板。

### 6.2 LVD 局部几何接口

对参考 Token $j_n^*$ 与竞争 Token $k$，使用固定 $W_0,b_0$ 定义：

$$
\hat u_{j_n^*k}=\frac{W_{0,k}-W_{0,j_n^*}}{\|W_{0,k}-W_{0,j_n^*}\|_2}
$$

$$
d_{n,k}(h)=\frac{(W_{0,j_n^*}-W_{0,k})^\top h+(b_{0,j_n^*}-b_{0,k})}{\|W_{0,j_n^*}-W_{0,k}\|_2}
$$

需要计算：

- 基准边际 $d_{n,k}^{(0)}=d_{n,k}(h_{0,n})$；
- 当前边际 $d_{n,k}^{(new)}=d_{n,k}(h_{\theta,n})$；
- 边际侵蚀 $d_{n,k}^{(0)}-d_{n,k}^{(new)}$；
- 无量纲侵蚀比例：

$$
x_{n,k}=\frac{d_{n,k}^{(0)}-d_{n,k}^{(new)}}{\gamma_0d_{n,k}^{(0)}}.
$$

实现要求：

1. 所有分母增加受控 epsilon，并记录发生近零分母的样本比例。
2. 理论上由基座 argmax 定义的 $d_{n,k}^{(0)}$ 应非负；小幅数值负值可仅在计算分母时替换为预注册的 $d_{min}>0$，同时保留原值用于日志。低于数值容差的负值应 fail-fast，因为这表示参考 Token、权重方向或缓存不一致。
3. 对 $d_{n,k}^{(0)}$ 极小的边界单独标记，不可静默 clamp 后与普通样本混合解释；测试必须覆盖近零边际和非整数 $p$。
4. 支持无 bias 的模型，将 $b_0$ 视为零向量但不实际分配完整词表大小的零张量。
5. 不生成 $V\times V$ 的 Token 对矩阵，只 gather 当前批次需要的权重行。
6. 默认按当前 Unembedding logits 排名选择参考 Token 以外的 Top-K，排名 detach；发生 argmax 翻转时必须包含当前赢家。均匀无放回随机采样保留为显式消融。
7. 默认 $K=512$，由点击入口的 `LVD_DIRECTION` 覆盖；不同 K/排名策略复用同一位置协议的全词表 d0，但必须重新校准 lambda。
8. 默认动态检查所有有效 response 预测位置，排除 prompt/padding/EOS，并保留 `base_correct` 筛选；切换位置协议需重建 d0，不得静默复用四位置缓存。按位置分块并在 backward 重算，不能通过平均分块 loss 改变全局二阶矩。
9. 新方法 `target_plus_anchor_ce_lvd` 同时使用 Target CE、`anchor_weight` 加权 Anchor CE 和 lambda 加权 LVD；同一 Anchor batch、单次合并 forward/backward。lambda 校准只比较 LVD 与 Target CE，不混入 Anchor CE。

### 6.2.1 LVD 代数 oracle 与模型适配验证

由于 Geometric Lens 快照未实现局部超平面与 margin 公式，本项目建立两个相互独立的 oracle：

1. **Logit-difference oracle**：直接使用模型线性头输出验证

$$
z_{j_n^*}(h)-z_k(h)=\|W_{0,j_n^*}-W_{0,k}\|_2\,d_{n,k}(h).
$$

2. **Laguerre-distance oracle**：显式构造小词表站点

$$
c_j=\frac{1}{2}W_{0,j},\qquad
\nu_j=b_{0,j}+\frac{1}{4}\|W_{0,j}\|_2^2,
$$

并验证

$$
\|h-c_j\|_2^2-\nu_j=\|h\|_2^2-z_j(h),
$$

因此最小 Laguerre 距离 Token、最大线性 Logit Token和所有成对边界符号完全一致。

项目必须提供真实模型验证动作。对捕获的每个 Unembedding 输入 $h$，先由 Laguerre power
argmin 独立确定候选胞腔 $j$，再显式验证凸多面体

$$
C_j=\bigcap_{k\ne j}
\left\{h:(W_j-W_k)^\top h+b_j-b_k\ge 0\right\}.
$$

验证必须扫描该胞腔对完整词表的 $V-1$ 个半空间，而非随机 K 或 Logit Top-K，并报告
Unembedding argmax token、LVD 胞腔 token、违规平面数、最小有符号平面距离、精确边界并列数
和实际检查总数。两种 token 不一致、任一半空间违规或点恰处于多个胞腔的精确公共边界时均
不得声称唯一对应关系成立。词表分块可以控制内存，但不得改变检查结果。

每个模型适配器必须分别在随机张量和真实捕获的 $h_0$ 上通过两类 oracle，并覆盖有/无 bias、tied/untied weights、padding 后多 Token 位置和混合精度输入。Geometric Lens 只用于人工核对最终 norm + `lm_head` 的读出顺序，不作为上述公式的唯一 oracle。

### 6.3 四段压强响应函数

实现文档定义的 $C^1$ 连续响应函数：

$$
g(x)=
\begin{cases}
0, & x\le 0,\\
x, & 0<x<1,\\
1+\dfrac{e^{\mu_3(x-1)}-1}{\mu_3}, & 1\le x<x_{bound},\\
g_3(x_{bound})+\dfrac{M}{\mu_4}\left(e^{\mu_4(x-x_{bound})}-1\right), & x\ge x_{bound},
\end{cases}
$$

其中 $x_{bound}=1/\gamma_0$、$\mu_4>\mu_3>0$、$M=e^{\mu_3(x_{bound}-1)}$。

初始默认值为 $\gamma_0=0.2$、$\mu_3=1$、$\mu_4=2$；容量权重默认 $p=1$，指数输入 clamp 上限默认为 20。所有参数均可由命令行/配置覆盖。

实现要求：

- 单元测试验证 $x=1$ 和 $x=x_{bound}$ 处函数值及一阶导连续；
- 在 FP32 中计算指数；
- 为防止混合精度溢出，提供可配置 exponent clamp；
- 记录进入四个区间的样本比例、clamp 次数和最大未截断指数输入；
- 使用有限差分与 PyTorch Autograd 对照梯度。
- 冻结分类头应冻结参数而非切断输入梯度；测试必须断言 $g(x)$ 及所有基于固定 $W_0,b_0$ 的损失对 $h_\theta$ 和至少一个可训练模型参数产生有限非零梯度。禁止在需要向隐藏状态回传的固定线性头 `forward` 上使用 `torch.no_grad()`。

### 6.4 容量权重和压强向量

容量敏感度权重定义为：

$$
\alpha_{n,k}=\frac{(d_{n,k}^{(0)}+\epsilon_d)^{-p}}
{\sum_{i\in N_K(j_n^*)}(d_{n,i}^{(0)}+\epsilon_d)^{-p}},
\qquad \sum_k\alpha_{n,k}=1.
$$

实际计算容量权重时使用经过上述规则保护的非负安全边际，原始边际仍单独记录。实现必须验证权重非负、有限且和为 1；任何 NaN/Inf 或显著负基准边际不得通过归一化被掩盖。

容量加权压强向量定义为：

$$
\widetilde{\Delta h}_n=
\sum_{k\in N_K(j_n^*)}g(x_{n,k})\alpha_{n,k}\hat u_{j_n^*k}.
$$

实现必须提供以下消融开关：

- 无容量权重，即竞争边界均匀加权；
- 只使用线性/Softplus/Hinge 响应，替换四段函数；
- 只惩罚最大危险边界；
- 随机 K 的取值与采样 seed；
- 使用原始位移 $\Delta h_n$ 替代压强向量。

### 6.5 统一协方差损失

令批次压强矩阵 $P\in\mathbb{R}^{N\times m}$ 的第 $n$ 行是 $\widetilde{\Delta h}_n^\top$，则：

$$
\Sigma=\frac{1}{N}P^\top P,
\qquad
L_{Unified-Cov}=\|\Sigma\|_F^2.
$$

这里的 $\Sigma$ 是未中心化二阶矩。主方法严格按该公式实现，并在变量与日志中使用 `uncentered_second_moment`。同时提供中心化消融：

$$
P_c=P-\mathbf{1}\bar p^\top,
\qquad
\Sigma_c=\frac{1}{N}P_c^\top P_c,
\qquad
L_{Centered-Cov}=\|\Sigma_c\|_F^2.
$$

中心化版本会移除批次平均压强，其物理含义与主方法不同，结果必须分开命名和解释。

不能默认显式构造 $m\times m$ 协方差矩阵。利用恒等式：

$$
\left\|\frac{1}{N}P^\top P\right\|_F^2
=\frac{1}{N^2}\|PP^\top\|_F^2,
$$

在 $N\ll m$ 时通过样本 Gram 矩阵计算完全等价的损失，以降低显存和计算开销。实现需根据 $N,m$ 自动选择较小路径，并用单元测试验证两种路径的前向值和梯度一致。

总损失为：

$$
L_{total}=L_{SFT}(D_T)+\lambda L_{Unified-Cov}(D_P).
$$

$\lambda$ 不使用跨模型硬编码默认数值：先运行 target-only pilot，以辅助损失对共享可训练参数的梯度范数约为 SFT 梯度范数的 50% 作为初始化校准目标，再在 $D_P^{cal}$ 上选择并固定最终值。

梯度比例校准使用固定、版本化的目标批次与 Anchor 批次，对 $L_{SFT}$ 和未乘 $\lambda$ 的 $L_{Unified-Cov}$ 分别计算相对于两者共享的全部可训练参数的全局 $L_2$ 梯度范数。DDP/FSDP 下先对各 rank 本地平方和进行 `all_reduce(SUM)` 再开方；梯度累积关闭，使用一次独立 FP32 校准 forward/backward，不更新 optimizer。至少在 8 个校准 batch 上取梯度范数比值的中位数，以 $\lambda_0=0.5/\operatorname{median}(\|g_{aux}\|/(\|g_{sft}\|+\epsilon))$ 初始化搜索；记录零梯度、非有限值、参数集合、聚合后范数及加权后的实际梯度比。若全参数校准在 FSDP 下不可行，必须采用有文档的固定参数子集并对所有方法保持一致，不能静默改口径。

### 6.6 分布式与数值要求

1. 正式实现中，$N$ 定义为当前一次 forward 在所有 rank 上的有效锚定位置总数，不跨 gradient accumulation micro-step；跨 micro-step 聚合会改变交叉项并要求保留计算图，除非另有经过验证的专门实现。
2. 多卡时使用支持梯度的 all-gather 或经过验证的等价分布式算法，不能将其他 rank 的压强向量 detach 后当常量。
3. 提供精确的分块 gather/分块 Gram 路径，避免一次物化超大全局 $P$ 或 Gram 矩阵；分块路径必须与朴素路径的前向值和梯度对齐，不能在显存不足时静默降级为单卡局部损失。
4. 正确处理每张卡有效锚定位置数不同的 padding 和 mask。
5. SFT loss 与辅助损失分别记录未缩放值、加权值和梯度范数。
6. 支持 $\lambda$ warmup，避免训练初期指数压强支配目标任务梯度。
7. 对 NaN/Inf、指数饱和、零边际和异常梯度提供 fail-fast 或明确告警。

### 6.7 诊断指标

每个评测周期至少记录：

- $d^{(0)}$、$d^{(new)}$ 和 $x$ 的均值及分位数；
- safety-core violation rate：$x\ge 1$；
- crossing rate：$d^{(new)}\le 0$；
- 四段 $g(x)$ 的占比；
- 压强向量模长；
- $\operatorname{Tr}(\Sigma)$ 与 $\|\Sigma\|_F^2$；
- 最大特征值、谱有效秩；
- 归一化各向异性，例如基于 $\Sigma/(\operatorname{Tr}(\Sigma)+\epsilon)$ 的指标；
- $\|h_\theta-h_0\|_2$；
- 随机竞争 Token 的累计唯一词表覆盖率；
- 完整词表 crossing 诊断及随机样本对真实 crossing competitor 的命中率；
- 新损失带来的时间和峰值显存开销。

完整词表 crossing 在固定诊断子集和每个正式评测 checkpoint 上通过完整词表 Logit 精确计算；同时保存诊断子集版本、随机采样 seed、累计唯一 token 数、有效翻转位置数和 Wilson/Bootstrap 置信区间，避免不同 run 使用不同样本口径。

#### 6.7.1 SFT 后 token 方向 interference 检验

所有 SFT 方法必须支持同一离线检验。令 $\Delta h_n=h_{\theta,n}-h_{0,n}$，其中 $h_{0,n}$
来自绑定同一 Anchor 样本与位置的 schema-v5 all-reference cache，$h_{\theta,n}$ 来自待检验
checkpoint。以固定基座 Unembedding 单位行 $\hat W_{0,k}$ 作为 token 方向，主统计为

$$
m_k^+=N^{-1}\sum_n\max(0,\hat W_{0,k}^{\top}\Delta h_n).
$$

正值表示 activation 朝 token $k$ 的读出方向移动；负投影表示远离，不能计入“朝该 token
移动”。产物必须覆盖完整词表，按 $m_k^+$ 降序提供 CSV 与带坐标轴的曲线图，并保存每个 token
的 signed mean、absolute mean、RMS 与 maximum absolute projection。汇总至少包含
$\sum_km_k^+$、absolute movement 总和、平均 $\lVert\Delta h\rVert_2$ 和 Top 1%/5%/10%
正向移动占比。不同方法只有在模型 revision、Anchor 指纹、位置配置和 d0 cache 完全一致时
才可比较；排序曲线本身被强制单调，局部 interference 集中应结合头部占比而非仅凭“单峰”外观判断。

#### 6.7.2 SFT 前后逐 token crossing 检验

该检验不使用 d0 cache。基座模型与待检验 checkpoint 必须在完全相同的 Anchor response token
流上运行。对每个受监督 label token，输出 argmax 与 label 不同记为 mismatch；同时报告基座
mismatch、微调后 mismatch、基座正确而微调后错误的新增 crossing、基座错误而微调后正确的
恢复、前后 argmax 改变次数和净 mismatch 增量。主比较项是 `new_crossing_count`、
`new_crossing_rate_among_base_correct` 与 `tuned_mismatch_count`，不能把基座原有错误都归因于 SFT。

### 6.8 正确性测试

1. Toy 词表上局部超平面与显式全词表 LVD 判定一致。
2. $d^{(0)}-d^{(new)}=\hat u^\top\Delta h$ 在符号约定下成立。
3. $h_\theta=h_0$ 时 $x=0$、压强为零、辅助损失为零。
4. 人工沿危险法向移动时，边际单调减小且损失单调增大。
5. 人工沿胞腔内部方向移动时，不产生错误正压强。
6. 四段函数连续性和梯度测试通过。
7. 显式协方差路径与 Gram 路径的值和梯度一致。
8. `lambda=0` 与普通训练模板对齐。
9. 至少一次多卡梯度一致性测试通过。
10. $c,\nu$ Laguerre oracle、Logit-difference oracle 与真实模型输出三方一致。
11. 捕获的最终隐藏状态恰好是实际 Unembedding 输入；逐层轨迹测试断言层数、索引和末层不重复，防止参考 Geometric Lens 中末层 hidden 复用一类错误。
12. 所有配置开关都有行为测试；若某损失权重非零但对应路径未执行、梯度恒为零或代码被禁用，测试必须失败。

### 6.9 实验顺序

1. Toy 几何测试；
2. 小模型 + 小数据，验证损失下降和梯度稳定；
3. 小模型完整 benchmark，搜索 $K,\gamma_0,\mu_3,\mu_4,p,\lambda$；
4. 3B 模型验证计算与效果；
5. 7B/8B 主实验；
6. 14B 或更大模型的 scaling 实验；
7. LoRA、量化和不同随机 K 的扩展实验。

### 6.10 阶段验收

- 所有几何和梯度单元测试通过；
- 小模型训练无持续 NaN/Inf；
- 相比 `target_only`，在至少 3 个随机种子的受控设置中达到预注册的 crossing/margin 改善门槛 $r_{min}$，并报告配对置信区间；
- 目标任务主指标退化不超过预注册的 $\delta_{max}$，不能通过完全阻止模型学习来获得保护结果；
- 随机邻居覆盖率和完整词表 crossing 诊断达到预注册门槛，否则只能将结果表述为随机局部近似下的改善；
- 给出时间、显存和缓存开销；
- 给出总压强下降与归一化各向异性变化，区分“整体缩小”和“方向均分”。

阶段验收的量化门槛统一引用第 4.4.4 节，不在本节复制数值；任何协议变更必须同步更新第 4.4.4 节的版本和时间戳。

### 6.11 阶段产物

- Unembedding 模型适配器；
- 全词表 d0 缓存与随机 K 邻居模块；
- 四段压强、容量权重和统一协方差损失模块；
- 参考缓存读取器；
- 训练插件与配置；
- 单元/集成测试；
- 小模型方法有效性报告。

---

## 7. 第四步：对比实验记录与分析框架

### 7.1 目标

形成可扩展的实验注册、聚合、统计检验、可视化和论文表格生成系统，保证所有 baseline 与主方法使用一致的数据、训练预算和评测协议。

### 7.2 实验注册表

每个 run 使用唯一 `run_id`，至少关联：

- 方法和方法版本；
- 模型 ID/revision；
- 目标任务、锚定集和评测集版本；
- seed；
- 超参数；
- Git commit；
- checkpoint 路径；
- 运行状态与失败原因；
- 硬件和 wall-clock time；
- 聚合指标文件和 per-sample 结果文件。

需要支持按“模型 × 目标任务 × 方法 × seed × 精度/PEFT 设置”生成实验矩阵，并检测缺失或重复 run。

### 7.3 Baseline 分层

为避免一开始同时复现过多方法，按以下优先级推进：

#### Tier 0：必须完成

- Base；
- Full FT target only；
- 冻结头主体微调 target only；
- LoRA target only；
- Target + Anchor CE；
- Target + Anchor KL；
- 本项目 LVD Unified-Cov Loss。

#### Tier 1：核心竞争方法

- SMART 或其他 data mixture；
- 一种 gradient orthogonal 方法，如 PCGrad；
- 一种专门面向 LoRA/持续学习冲突的方法，如 LOTA、SLORA 或 LoRI；
- 本方法关键消融。

#### Tier 2：扩展对比

- DARE/DAREx/DiFT 等 delta parameter pruning；
- KnOTS/TSV-Merge/PCB-Merging 等 model merging；
- CLUE/CircuitLoRA/FLU 等可解释性方法；
- DYNAMIXSFT、LIMA 等其他数据策略。

Tier 2 只有在主方法闭环稳定后开始，且必须区分“训练时保护”“训练后修复”和“模型合并”，避免把不同问题设定直接混为同一组排序。

### 7.4 公平比较协议

1. 同一模型 revision、数据 split、prompt、token budget 和评测配置。
2. 对学习率、正则权重等允许每种方法在相同校准预算内调参。
3. 同时报告训练样本数、训练 Token 数和总 FLOPs/时间。
4. 至少 3 个随机种子，报告均值、标准差和置信区间。
5. 优先使用 per-sample paired bootstrap 或适合任务指标的配对检验。
6. 不只比较最高通用分数，应绘制目标任务收益与通用能力保留的 Pareto 前沿。
7. 失败运行和排除规则必须保留并说明，不能只保存成功结果。

### 7.5 分析功能

1. 主结果表：目标任务效果、通用能力变化、平均保留率。
2. Scaling：模型大小与保护效果、额外计算开销的关系。
3. 消融：响应函数、容量权重、协方差形式、随机 K 大小和三个设计目标。
4. 内部指标：
	- LVD margin 分布；
	- safety-core violation 和 crossing；
	- 压强协方差谱；
	- 归一化各向异性；
	- 激活漂移；
	- 边缘 Token 子集表现。
5. 稳定性：loss 曲线、梯度范数、NaN/Inf、指数 clamp 比例。
6. 效率：吞吐量、每步时延、峰值显存、缓存大小。
7. PEFT/量化：LoRA、QLoRA 或量化模型下的效果与误差。
8. 自动生成 CSV/Parquet 汇总、LaTeX 表格和固定风格图表。

### 7.5.1 局部干扰归因诊断（可选但优先于复杂 baseline）

为直接检验“通用能力冲突来自少量 feature/组件的局部高 interference”，在主方法闭环后增加 checkpoint 级离线诊断。其设计借鉴 Future Localization 的 clean/corrupted activation-gradient 乘积，但独立实现并与训练循环隔离。

1. 固定同一批锚定样本和 Token 位置，比较 $M_0$、`target_only` 和本方法 checkpoint。
2. `clean/corrupted` 必须在实验前定义。首选成对语义扰动或 Base/SFT 状态对照；不得随意把不同长度样本配对后只按位置截断。
3. 诊断 loss 优先使用 LVD margin erosion 或参考 Token 与真实竞争 Token 的 logit difference，而不仅是通用 task loss。
4. Hook 捕获目标从组件级开始：attention 的 q/k/v/o projection 和 MLP gate/up/down projection。只有组件级结果支持局部集中假设时才扩展到 head/神经元/边。
5. 对每个组件保存 raw、absolute、mean 和 $\sqrt{numel}$ 归一化 score、Token 数、元素数、层号和模块类型，防止大矩阵因元素多而天然排名靠前。
6. 报告局部集中度：Top-1%/5%/10% score mass、Gini/HHI、层间分布、跨 seed 排名稳定性，以及归因热点与 LVD crossing 样本的相关性。
7. 主方法若有效，应同时验证 crossing/margin 改善和 attribution concentration/热点稳定性的变化；归因只作机制证据，不替代任务指标。
8. 诊断默认只在保存的 checkpoint、固定小子集和单独进程运行，不进入每个训练 step，不写全参数 dense mask，不默认计算 HVP 或导出完整 circuit graph。
9. 输出使用稀疏 Top-N 记录或组件级表格。若生成激活/梯度缓存，记录文件数、字节数、写入时长和清理策略，超过预算时分片或缩小诊断集，不能静默丢弃组件。
10. 诊断运行前后比较模型 train/eval 模式、`requires_grad`、optimizer 参数组和 `.grad`，确保没有状态污染。

二维 LVD 图只作说明性图形：从同一固定样本选择少量 Token/轨迹，投影方法、随机种子和拟合数据必须记录，并在图注中声明二维区域不是高维胞腔的精确几何。首版不复用 Geometric Lens 的 Autoencoder；若后续引入，必须先修复并测试冻结分类头的输入梯度、邻域损失真实执行、无 bias 支持和层轨迹索引。

### 7.5.2 正式评测编排

通用能力评测以 Future Localization 中的 `lm-evaluation-harness` 快照为来源基线，在独立工具目录恢复为完整的官方 0.4.9 工作树后，由统一 orchestrator 调用其 Python API 或 CLI。原始裁剪快照保持只读，恢复差异和后续必要补丁单独记录；评测源码不复制进主训练包。

每次运行保存：

- 完整命令/调用参数、请求任务列表、实际完成任务列表与 harness commit/version/worktree hash；
- 模型 checkpoint/revision、Tokenizer revision 和 chat template hash；
- few-shot、batch size、seed、generation kwargs 与 dtype；
- 任务 YAML/config hash、filter、metric、bootstrap 配置和 request cache key；
- harness 原始 JSON、per-sample 输出、stderr/stdout、退出码和本项目标准化结果；
- 失败任务、重试次数与运行耗时。

Orchestrator 必须比较请求任务和实际结果任务集合，任何任务缺失都使整个 run 失败。标准化层只重命名字段和附加元数据，不重新计算或覆盖 harness 原始指标。LoRA checkpoint 需明确使用 Adapter 加载还是 merge 后模型，并在小子集验证两种读出方式结果一致。

训练期 eval loss/perplexity 只用于健康监控，不得替代任务原生 benchmark。

### 7.6 阶段验收

- 任意 run 可通过 `run_id` 追溯到配置、代码、模型和数据版本；
- 同一分析脚本可聚合所有方法，不依赖手工复制结果；
- 自动检查实验矩阵缺失项和 seed 数量；
- 主结果表、Pareto 图、边际/越界图、谱图和效率表可一键生成；
- 所有论文数字能追溯到 per-run 结构化结果。

### 7.7 阶段产物

- 实验注册与调度配置；
- 统一结果 schema；
- 聚合和统计检验模块；
- 图表与 LaTeX 表格生成脚本；
- 主实验、消融、scaling、稳定性和效率报告。

---

## 8. 建议代码结构

```text
SFT-Conflict/
├── configs/
│   ├── model/
│   ├── data/
│   ├── training/
│   ├── method/
│   └── experiment/
├── src/sft_conflict/
│   ├── cli/
│   ├── config/
│   ├── data/
│   ├── models/
│   ├── adapters/
│   ├── trainers/
│   ├── geometry/
│   ├── losses/
│   ├── evaluation/
│   ├── callbacks/
│   ├── tracking/
│   └── analysis/
├── scripts/
│   ├── bootstrap/
│   ├── cache/
│   ├── train/
│   ├── evaluate/
│   └── analyze/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── distributed/
├── docs/
├── REQUIREMENTS.md
└── pyproject.toml
```

模块职责与 Future Localization 蓝本建立明确映射：

- `config/` 取代 Fastargs，保存 typed schema 和旧配置迁移器；
- `models/` 取代 `model/finetuning.py` 的模型加载部分，负责离线加载、精度、Unembedding 和 tied weights；
- `adapters/` 只包含 standard PEFT LoRA；
- `data/` 重写 `FinetuningDataset`/collator 为目标与 Anchor 双流；
- `trainers/` 继承 Future Localization 的 `BaseFinetuningTrainer`/`TargetFT*` 思路，但只保留一个 `SFTConflictTrainer`；
- `losses/` 保存纯函数和插件，不依赖 Trainer、mask 或可解释性模块；
- `evaluation/` 分离目标任务 evaluator 与 lm-eval bridge；
- `callbacks/` 负责结构化日志、冻结状态断言、容量预检和 checkpoint retention。

模型权重、数据集、参考激活缓存、checkpoint、恢复后的 lm-eval 工具工作树和实验输出均放在 Git 忽略目录或外部存储中，不提交 GitHub。公开仓库 URL、revision、配置、小型测试 fixture 和必要的第三方补丁可以提交。

---

## 9. 非功能性要求

### 9.1 可复现性

- 所有实验由配置驱动，不依赖手工修改源码；
- 固定并记录 Python 和核心依赖版本；
- 支持 seed 设置，但同时记录无法完全确定化的 CUDA 算子；
- 数据、模型和代码均使用不可变 revision 或内容哈希。

### 9.2 可测试性

- 几何公式、响应函数、协方差等核心逻辑必须是无训练框架依赖的纯 PyTorch 函数；
- 单元测试优先使用小张量和 Toy Vocabulary；
- 至少提供单卡短训练和多卡短训练集成测试；
- 任何性能优化必须与朴素参考实现进行数值对照。
- 关键固定模块（冻结 Unembedding、冻结参考模型）必须额外验证“参数无梯度/不更新”与“输入梯度可传播”这两个不同条件，不能用 `no_grad` 代替参数冻结。

### 9.3 安全与数据管理

- Hugging Face、W&B 和其他平台 Token 不得写入配置、日志或 Git；
- 代码执行评测必须隔离；
- 明确数据集许可证和允许的使用范围；
- checkpoint、缓存和日志设置保留与清理策略。
- 维护 `THIRD_PARTY_NOTICES.md` 或等价机器可读清单，记录复制代码的来源 URL、commit/snapshot hash、文件映射和许可证；按具体来源保留 MIT 版权声明、Apache-2.0 文件头，并在上游存在 NOTICE 文件时传递其要求保留的内容。
- 所有 Hub 模型和数据只从固定 revision 的本地 cache 加载；训练与评测进程默认开启 Hugging Face/Transformers/Datasets 离线模式。

### 9.4 性能

- 首版优先保证公式正确，再优化吞吐量；
- 避免全词表 Token 对、完整胞腔和不必要的 $m\times m$ 矩阵；
- 为参考模型在线前向、缓存读取、当前模型前向和几何损失分别计时；
- 对 Anchor Token 数量、词表大小、随机 K 和缓存精度提供显存/存储估算器。
- 每次正式运行前检查 `/data1` 可用容量；低于 2 TB 或不足预计新增量 2 倍时停止并反馈。

---

## 10. 需要用户提供或确认的参考代码与信息

第一步的工程基线已经确认。开始实施时按下表处理剩余的外部输入和执行性条件。

| 输入资料 | 优先级 | 是否阻断 | 缺失时的处理方式 |
|---|---|---|---|
| GPU 与存储 | 已确认 | 不阻断 | 4×A100 40GB；Artifact 位于 `/data1/chenhang/SFT_Conflict/artifacts`，当前约 11 TB 可用 |
| 模型与目标任务 | 已确认 | 不阻断 | Qwen2.5-0.5B 冒烟、3B 正式、7B 扩展；SST-2/GSM8K 冒烟，GSM8K 正式 |
| Hugging Face cache | 路径与离线协议已确认，内容尚未准备 | 阻断真实模型/数据运行 | 通过可用下载通道或迁移方式填充固定 revision cache |
| Future Localization Full FT/LoRA 蓝本 | 已提供完整 `src` 快照 | 不阻断 | 按第 4 节裁剪重构，不再等待额外训练模板 |
| LVD/Geometric Lens 参考实现 | 已提供本地快照 | 不阻断 | 只复用读出/可视化思路，LVD loss 独立实现 |
| 通用能力评测工具 | 已提供 lm-eval 裁剪快照 | 阻断正式通用评测 | 从官方 0.4.9 恢复完整工作树并通过兼容性/对齐测试 |
| 其他数据清洗与专项评测脚本 | 第二步前建议 | 不阻断 | 固定公开官方实现及 commit |
| SMART 与可解释性参考实现 | 已提供本地快照 | 不阻断 | 分别用于数据血缘和离线诊断设计 |
| 方法 Notebook、伪代码和超参数经验 | 可选 | 不阻断 | 使用 Toy Test 和校准集确定 |
| Baseline 官方仓库与运行配置 | 第四步前建议 | 不阻断主方法 | 按 Tier 顺序逐项引入 |

### 10.1 最高优先级：开始第一步前

#### 1. 已提供的 Future Localization 微调蓝本

`Future_localization-master/src` 已补齐，足以支持第一步设计，不再要求用户额外提供 Full FT/LoRA 模板。项目使用的核心参考是：

- `src/exec/finetuning_model.py`：配置分区与入口；
- `src/model/finetuning.py`：模型、数据、训练、保存和评测编排；
- `src/finetuning/base.py` 与 `FT.py`：目标/通用 CE/KL 损失分派；
- `src/model/lora_utils.py`：仅参考 standard PEFT LoRA 分支；
- `src/dataset/Base.py`：参考结构化 target/pervasiveness batch；
- `src/training_losses.py`：参考标签清洗、Causal LM loss 和 KL 纯函数。

这些文件是行为蓝本而非待复制实现；第 4.2 节列出的解释性功能和旧训练循环不进入新项目。若后续提供其他已跑通模板，只用于交叉验证，不改变当前主架构。

#### 2. 三份参考快照的来源信息

代码内容已经足够支持需求设计。若方便，请补充三个压缩包对应的上游 URL、branch/tag/commit，以及是否包含你个人未提交的修改。缺少这些信息不阻断开发，项目会使用 snapshot hash 固定当前版本。

Geometric Lens 中目前可用于人工对照的范围是：

- 最终 LayerNorm + `lm_head` 读出顺序；
- Token Top-K/排名；
- 二维决策区域与轨迹的展示构图。

它没有实现本项目的 LVD margin loss，且 Autoencoder/hidden trajectory 存在需要独立修复验证的路径，因此不再要求用户额外提供“LVD 构建模板”。

#### 3. 已确认的服务器与训练资源

当前服务器为单机最多 4×NVIDIA A100-SXM4-40GB，Driver 535.129.03，驱动报告 CUDA 12.2。`/data1` 当前约有 11 TB 可用空间，Artifact 根目录固定为 `/data1/chenhang/SFT_Conflict/artifacts`。未检测到可用 Hugging Face cache，且现有本地代理不可用；真实模型实验前需要先完成 cache 下载或迁入。未提供 Slurm 信息，因此首版按本机直接启动设计，同时保持 launch command 可迁移。

#### 4. 已确认的第一轮模型与任务

- 冒烟模型：`Qwen/Qwen2.5-0.5B` Base；
- 首轮正式模型：`Qwen/Qwen2.5-3B` Base，之后扩展 7B；
- 冒烟目标任务：SST-2 与 GSM8K，各 256 条；
- 正式目标任务：GSM8K；
- Anchor 数据：指定 SMART FLAN mixture，规模依次为 256/5k/20k；
- lm-eval 冒烟：HellaSwag 与 GSM8K 限制集；正式使用已固定的 9 项任务组。

### 10.2 第二步前

#### 5. 数据与评测参考代码

通用能力评测默认使用已提供的 lm-evaluation-harness 快照及其环境中指定的 `lm-eval==0.4.9`，无需再提供另一套通用评测框架。若知道该裁剪快照的上游 commit 或生成方式，请补充；不知道也不阻断恢复，将以官方 0.4.9、当前 snapshot hash 和差异清单共同固定版本。

仍请提供已有的数据清洗、prompt template 和专项评测脚本，尤其是论文计划中实际采用的版本，例如：

- SMART 数据选择结果或 SMART 官方仓库；
- 对第一版通用任务组的 few-shot/chat-template 偏好；
- GSM8K/MMLU/BBH/CommonsenseQA 的补充任务配置或期望评测协议；
- HumanEval/MBPP 的安全执行框架；
- AlpacaEval 的版本与 judge 配置。

已确认当前没有下载好的模型和数据 cache；统一 cache 路径及离线加载协议见第 4.4.1 节。准备 cache 时如需访问令牌，应直接在服务器的下载环境中登录，不通过聊天、配置文件或 Git 传递。

### 10.3 第三步前

#### 6. 方法实现偏好或原型

如果已有 Notebook、伪代码或局部实现，请提供：

- $h_0$ 与 $h_{new}$ 的抽取位置；
- 一个“样本”对应最后 Token、全部 Token 还是指定 Token；
- 随机 K 的初始建议；
- $\gamma_0,\mu_3,\mu_4,p,\lambda,\epsilon_d$ 的建议范围；
- `lm_head` 冻结策略；
- 是否已有容量权重或协方差代码；
- 是否计划在线参考模型还是预计算缓存。

### 10.4 第四步前

#### 7. Baseline 优先级和官方实现

不需要现在一次性提供全部 baseline。主方法闭环后，请按优先级提供准备复现的官方仓库、commit 和已知运行配置，优先选择：

- 一种 data mixture；
- 一种 gradient orthogonal；
- 一种 LoRA/持续学习冲突方法；
- 一种 parameter pruning 或 model merging；
- 一种 interpretability 方法。

若已有论文表格模板、W&B 项目、结果 CSV schema 或绘图代码，也应一并提供，以减少第四阶段迁移成本。

---

## 11. 总体里程碑与进入条件

| 里程碑 | 完成标志 | 下一阶段进入条件 |
|---|---|---|
| M1 训练底座 | Future Localization 蓝本完成裁剪重构；Full FT、冻结头主体微调、standard LoRA、双流接口、离线 cache、lm-eval 与必要模型/LVD 检查可运行 | 小数据过拟合、断点恢复、单/多卡和模型适配测试通过 |
| M2 数据与基线 | 数据注册、评测、参考缓存和 benchmark 完成 | 至少一个目标任务被有效学习 |
| M3 方法闭环 | LVD Loss 接入，小模型有效性与稳定性验证完成 | crossing/margin 指标改善且目标任务未失效 |
| M4 完整实验 | 多模型、多任务、baseline、消融和统计分析完成 | 所有结果可追溯并自动生成图表 |

任何阶段未达到验收条件时，应优先解决当前闭环，不提前扩大模型、数据集或 baseline 数量。

---

## 12. M1 实施检查清单

以下事项全部属于第一步和 M1 验收，不构成额外预备阶段。第 4.4 节已经固定工程基线：

1. 创建并锁定训练环境和独立 lm-eval 0.4.9 环境，验证 PyTorch CUDA/BF16；
2. 创建 Artifact 目录结构，并安装容量预检；
3. 通过届时可用的下载通道或文件迁移方式填充 Hugging Face cache，固定 Qwen2.5、GSM8K、SST-2 和 SMART FLAN dataset commit；
4. 恢复完整 lm-evaluation-harness 0.4.9 工作树并通过 smoke/parity test；
5. 为三份参考代码生成 snapshot hash；若用户后续提供上游 URL/commit，再补充来源映射；
6. 生成 Future Localization 蓝本到新模块的迁移矩阵，逐项标记“参考、重写、删除”，作为第一步代码评审清单。