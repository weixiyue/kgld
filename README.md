# KGLD

主模型由 OC-iVAE 更名为 KGLD：实现为 `models/kgld.py::KGLD`，主配置为 `config/kgld.yaml`。
更名不改变模型计算或 state_dict keys；旧 `model_type: oc_ivae` 和未写类型的配置仍可加载。
历史设计/审计文档、配置快照和实验产物保留原名；仓库目录仍为 oc-ivae-align。

使用 PyTorch 在 N-CMAPSS 上构建无 HI/RUL 标签监督的健康指标，提供 KGLD、cVAE、VAE，
以及采用 KGLD 数据协议的 OCALD、Proposal-C CNN 和 Residual-A/B 适配版。
原三模型使用 `train.py`，新模型使用独立实验入口；全部共用 `evaluate.py`。
项目规范见 [AGENTS.md](AGENTS.md)，当前状态见 [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)。

## 环境

本轮验证环境为 Python 3.11.15、PyTorch 2.5.1+cu121，实际检查均使用 CPU。
[requirements.txt](requirements.txt) 固定已验证的运行依赖；Torch 固定发行版本 2.5.1，
CUDA wheel 的构建需与目标环境匹配。PyTables (`tables`) 用于 pandas 的 cycle-HI HDF 读取。

```bash
python -m pip install -r requirements.txt
```

依赖安装命令仅供环境准备；重构过程中没有安装或升级包，也未验证 GPU。
精确数值复现应保持同一依赖版本、数据、checkpoint、线程和随机状态。

## 目录

```text
train.py                       # 训练 CLI 和共享普通函数
evaluate.py                    # cycle 推理、标签关联、dev-only 缩放与导出
dataset.py                     # flight/window Dataset、unit/cycle 划分与 sampler
models/
    kgld.py                 # 模型装配、分支、FiLM、conditional prior、HI head
    tcn.py                     # causal encoder 与 non-causal decoder
    vae.py                     # 两个已有 baseline 的差异组件
    ocald.py                   # 原 8D I/C OCALD、padding 适配及模型专属损失
    proposal_cnn.py            # Proposal-C CNN 与 batch Pearson 约束
    residual.py                # Residual-A/B、磁盘残差、窗口 PCA/方向与 flight 聚合
    __init__.py                # 显式模型选择
utils/
    preprocess.py             # N-CMAPSS 数据处理
    losses.py                 # 重构、KL、HSIC、ranking、soft Spearman
    metrics.py                # 独立指标公式
    experiment.py             # 新实验共用的配置快照、history 与 checkpoint 保存
    optimizers.py             # 迁移 CNN 共用的 Keras Adam 数值语义
config/                       # 全部独立实验 YAML
experiments/                  # 独立新模型训练入口及 W-summary 配对实验
tests/                        # 模型数学、采样、重载与端到端合成 smoke
data/raw/                     # 原有符号链接
data/processed/               # 可复用的预处理 HDF5，Git 忽略
runs/                         # 新运行配置、checkpoint、history 和评估结果
checkpoints/                  # 既有 checkpoint，原地保留
docs/                         # 当前说明、设计和历史验证资料
```

当前 `data/raw` 链接到 `/home/xiyue/CodeSpace/raw/`。保留链接及其目标，不递归移动、删除或复制原始数据。
当前可用的是 DS03 主文件与 dev/test cycle-HI 文件；DS02 主文件缺失。

## 数据预处理

可在训练前预处理一次：

```bash
python -m utils.preprocess --config config/kgld.yaml
```

结果保存为 `data/processed/<原文件名>_<处理标识>.h5`，包含 dev/test 的
`x`、`mask`、`W_seq`、`u`、unit/cycle、周期 RUL、归一化统计量和 `L_max`。
HI 参考仍独立读取，不写入模型输入缓存。预处理不使用模型参数，也不消耗训练随机状态。

未手动预处理时，训练或评估首次启动会提示并自动生成；之后相同数据和处理参数直接复用。
Dataset 只加载所选 unit 的已处理张量到内存，取样时不再重复归一化或构造 `u`。
训练、评估、不同模型及配对实验共享处理文件，train/val 的 seed 和比例不影响缓存标识。
首次生成同时准备 dev/test，test 不参与训练目标、模型选择或 dev 统计量计算。

处理标识包含原文件实际路径、大小、纳秒修改时间、降采样因子、`n_fc` 和预处理版本。
数据或参数变化生成新的文件，不覆盖旧文件；缓存损坏或不匹配会报错，不自动重建。
文件信息检查不扫描原始文件内容，无法识别刻意保持路径、大小和修改时间的内容替换。
预处理算法变化时需更新 `utils/preprocess.py` 的 `PROCESSED_VERSION`。
可在 YAML 的 `data.processed_dir` 指定另一处理目录；默认目录及生成数据均不提交 Git。
运行快照记录实际 `processed_data` 路径，仍需保留原文件以核对来源信息。

## 训练与评估

从仓库根目录运行。每个新训练必须使用不存在的目录；默认名称来自配置文件名。
既有同名历史 runs 不会被覆盖，使用 `--run-dir` 明确指定新名称。

```bash
python train.py --config config/kgld.yaml \
  --run-dir runs/kgld_new
```

训练按 validation reconstruction 选择最佳 checkpoint，再自动评估全部 dev 和 test。
HI 标签不进入训练目标，test 不参与模型选择。

独立重评使用新的输出目录；配置快照可定位同一运行的 `best.pt`：

```bash
python evaluate.py --config runs/kgld_new/config.yaml \
  --output-dir runs/kgld_new/reevaluation --device cpu --threads 2
```

旧 checkpoint 使用显式路径，无需修改或迁移权重：

```bash
python evaluate.py --config config/history/exp_tuning_mono_weight.yaml \
  --checkpoint checkpoints/exp_tuning_mono_weight/best_model.pt \
  --output-dir runs/exp_tuning_mono_weight_legacy_check --device cpu --threads 2
```

查询参数不会启动训练或推理：

```bash
python train.py --help
python evaluate.py --help
python -m experiments.run_w_summary_ablation --help
```

## 模型、配置和后续实验

| 模型 | 先验 | z_c 分支 | decoder W summary |
| --- | --- | --- | --- |
| KGLD | u 条件 Gaussian | FiLM(u) | 由现有配置开关控制 |
| cVAE | 标准正态 | 保留 FiLM(u) | 保留同一配置控制 |
| VAE | 标准正态 | 无 FiLM | 保留同一配置控制 |

当前 VAE 仍是双 latent 分支，不能解释为完全移除了 W/u 所有路径的单分支模型。
模型由 `experiment.baseline_type` 的缺省值、`cvae` 或 `vae` 显式选择。
现有 YAML 全部保持数值与独立实验身份，包括各自的早停和 loss 权重差异；没有额外统一或调参。
历史 `postprocess.ewma_alpha` 不应用到当前评估，部分 logging 字段保留原记录用途。

后续新增模型放在 `models/`，新增实验放在独立 `config/*.yaml`；
共享训练/评估部分继续调用普通函数。模型特有目标和 HI 构造按实际差异扩展，
不要求新模型伪造 KGLD 的 12 个字段，不提前引入 registry 或通用 Trainer。
批量脚本只调度、审计和汇总，不复制训练循环与指标计算。

## OCALD 与 Proposal-C

两者均直接复用公共 Dataset/预处理缓存、seed=42 的 train/val 设备划分和完整 dev/test 评价，
不使用 HI/RUL 训练或选模；不是源仓库的数据协议或完整实验复现。

| 配置 | 训练方式与保留项 |
| --- | --- |
| [ocald_original](config/history/ocald_original.yaml) | 表示 100 轮后冻结，只训练 HIHead 100 轮；I/C 各 8 维 |
| [ocald_joint](config/history/ocald_joint.yaml) | 联合 100 轮；KL 各 1、排序 10、头部 L2=1e-4 |
| [ocald_joint_decoder_w](config/history/ocald_joint_decoder_w.yaml) | 同上，显式开启 decoder W 输入 |
| [proposal_cnn_c](config/proposal_cnn_c.yaml) | CNN + batch Pearson C；Adam 1e-4、batch 20、50 轮 |
| [residual_a](config/history/residual_a.yaml) | 旧 flight 配置，已有训练数值保留 |
| [residual_b](config/history/residual_b.yaml) | 旧 flight 配置，已有训练数值保留 |
| [residual_a_short_window](config/residual_a_short_window.yaml) | X/W 重构，长度50/stride1；健康段窗口 MAE、20轮 |
| [residual_b_short_window](config/residual_b_short_window.yaml) | 仅 W 预测 X；相同窗口、训练与 flight 评价协议 |

实际迁移来源固定为 [kgld 4f9e53cd3623b5cf0311d594fd7efcd4bec88206](https://github.com/weixiyue/kgld/tree/4f9e53cd3623b5cf0311d594fd7efcd4bec88206)
的 model.py、losses.py、train.py 与对应三份配置，以及
[UnsupervisedHI 157ebfae7baeee1e0cd52180f8e58ead3dcb448e](https://github.com/KBaja/UnsupervisedHI/tree/157ebfae7baeee1e0cd52180f8e58ead3dcb448e)
的 CMAPSS/03_train_evaluate_proposals.ipynb、CMAPSS/eval_utils.py；每份 YAML 记录完整 commit 和文件。

OCALD 仅把有效 X/W 按原序移到前端、右补零，再送入 packed GRU；训练和推理共用该适配。
高斯重构按 flight 求和、KL 按维求和后取 batch 均值，排序仍为鼓励 score 随周期增加的 softplus。
保留源 Adam 1e-3、裁剪 5（HI 阶段不裁剪）、原均匀设备/配对及联合 4x4 分组采样。
默认 decoder 不读取 W；conditional prior 仍读取 W。公共评价才取 `-score(mu_I)`，训练不改符号。

Proposal-C 保留通道 [128,64,32,16]、kernel 85、same/ReLU，按 Keras 时间优先顺序 Flatten 得到 Z，
重复 Z 拼接逐时刻 W，再用反向通道序列卷积及逐时刻 Dense 输出 14 维。
平均 MAE 包含公共补零位置；C 是整个 batch 的 Pearson(Z, cycle)，权重 1，最小化鼓励负相关。
先顺序组 batch 再 shuffle batch，最后不足一个 batch 也保留；零方差导致非有限 C 时明确报错。
使用 Glorot uniform/零偏置；Keras Adam 显式 beta=(0.9,0.999)、epsilon=1e-7（uncorrected sqrt(v) 后的 epsilon-hat）。
源 Notebook 未指定 seed，此适配显式用 42；PyTorch RNG/浮点实现不保证与 TensorFlow 逐位相同。
源 eval_utils.py 的函数签名缺逗号，只迁移已核对算法，不复制其训练/评价工具。

```bash
python -m experiments.train_ocald --config config/history/ocald_original.yaml \
  --run-dir runs/ocald_original_seed42 --device cpu --threads 2
python -m experiments.train_ocald --config config/history/ocald_joint.yaml \
  --run-dir runs/ocald_joint_seed42 --device cpu --threads 2
python -m experiments.train_ocald --config config/history/ocald_joint_decoder_w.yaml \
  --run-dir runs/ocald_joint_decoder_w_seed42 --device cpu --threads 2
python -m experiments.train_proposal_cnn --config config/proposal_cnn_c.yaml \
  --run-dir runs/proposal_cnn_c_seed42 --device cpu --threads 2
python evaluate.py --config runs/proposal_cnn_c_seed42/config.yaml \
  --output-dir runs/proposal_cnn_c_seed42/reevaluation --device cpu --threads 2
```

新入口默认 CPU，可显式选择 cuda；每次必须使用新目录。加 `--smoke` 后每阶段 1 轮、最多 1 次更新，
OCALD 仅取各训练设备前 8 个 flight；公共 scaler 和七项指标始终使用完整 dev/test，不缩小评价集合。
新模型按最后轮保存 `best.pt`，元数据 selection=last_epoch；两阶段另存 `representation.pt`。
checkpoint 包含模型类型、阶段、完整实际配置、seed 和预处理出处。
重评使用运行快照，结构/数据不一致的外部配置会报错，其余设置优先使用 checkpoint 保存配置。
每轮 history 记录真实损失项、权重、优化步数、抽取样本数和耗时；CNN 不记录伪造的 KL/HSIC。

Residual-A/B 来自 UnsupervisedHI commit `157ebfae7baeee1e0cd52180f8e58ead3dcb448e` 的
`CMAPSS/01_data_preprocess.ipynb`、`pre_process_utils.py`、`02_train_evaluate.ipynb` 和 `eval_utils.py`。
A 为 concat(X,W)→4×Conv64/kernel11/tanh→逐时刻 Dense14；B 为 W→4×Conv64/kernel11/ReLU→Conv14/kernel11。
Glorot uniform/零偏置和 Keras Adam 数值约定与 Proposal-C 共用，网络与 MAE 未改动。

新增独立 `residual_a_short_window.yaml`、`residual_b_short_window.yaml`，显式 `data.input_mode: short_window`。
默认长度50、stride1是参考 Notebook 的短窗口候选，不声称适用于所有论文实验。
沿用 KGLD 原文件、变量顺序、设备划分、设备 X/dev 全局 W 标准化及每 flight 下采样。
变长缓存复用公共 flight 缓存的统计，存储 **L_max 截断及 padding 之前** 的真实序列；
Dataset 只保存 (flight, start) 索引并按需切片，不物化全部重叠窗口。缓存身份包含模式、窗口设置和处理规则。
窗口不跨 flight，数量为 `floor((L-window_length)/stride)+1`；过短 flight 报错并列出键。
公共 flight 缓存和新变长缓存均保留在 data/processed/，可用以下命令提前准备：

```bash
python -m utils.preprocess --config config/residual_a_short_window.yaml
```

网络仅更新 train 设备 cycle<=20 的窗口，不使用 hs/HI/RUL；样本级 shuffle 后组 batch。
新短窗口配置保持 batch512、lr1e-4、20轮、seed42、最后轮，无调度/早停或其他损失。
旧 `residual_a.yaml`/`residual_b.yaml` 保留 flight 输入和已有数值（当前200轮），旧 checkpoint/结果可继续重评。

窗口绝对残差按时间优先展平，用全部 dev **窗口** 拟合普通单分量 PCA；
按各 dev 设备 cycle/窗口起点排序后的前10与后10 **窗口** 均值差确定全局方向，负差时翻转。
test 复用 PCA 和方向；随后同 unit/cycle 的窗口分数取均值，才交给公共 dev 全局 Min–Max 和七项指标。
不跨周期平滑、不逐设备缩放、不裁剪。RUL/HI 仅在聚合后按 flight 关联，不按窗口重复计权。
残差分批写入临时磁盘矩阵，普通 PCA 使用 auto solver，不替换为 IncrementalPCA；临时矩阵在上下文退出时删除。
checkpoint 保存数值 PCA 参数、方向、完整 dev flight 键、窗口键摘要/数量、窗口设置及预处理来源，重评不重拟合。
这是采用 KGLD 归一化的短窗口适配版，**不是原论文完整数据协议复现**。

```bash
python -m experiments.train_residual --config config/residual_a_short_window.yaml \
  --run-dir runs/residual_a_short_window_seed42 --device cuda --threads 2
python -m experiments.train_residual --config config/residual_b_short_window.yaml \
  --run-dir runs/residual_b_short_window_seed42 --device cuda --threads 2
python evaluate.py --config runs/residual_a_short_window_seed42/config.yaml \
  --output-dir runs/residual_a_short_window_seed42/reevaluation --device cuda --threads 2
```

入口默认 CPU；加 `--device cpu --smoke` 只做1轮、最多1个健康段 batch，PCA/评价仍用完整 dev/test。
history 记录健康 flight 数、候选窗口数、更新步数/累计步数、实际抽样数、MAE 及耗时。
每次使用新目录，拒绝覆盖既有结果；旧 flight 模式按每设备前后10个 flight 定向，保持历史语义。

独立线性探测入口（使用参考 HI 标签的表征评价，**不属于公共 HI 七项指标**）：

```bash
python -m experiments.evaluate_linear_probe --config runs/ds03_default/config.yaml \
  --output-dir runs/ds03_default/linear_probe --device cuda --threads 2
python -m experiments.evaluate_linear_probe --config runs/ocald_original/config.yaml \
  --output-dir runs/ocald_original/linear_probe --device cuda --threads 2
python -m experiments.evaluate_linear_probe --config runs/proposal_cnn_c/config.yaml \
  --output-dir runs/proposal_cnn_c/linear_probe --device cuda --threads 2
```

默认读取运行快照对应的 best.pt，可用 `--checkpoint` 指定其他已确认权重；输出目录必须尚不存在。
可用 `--device cpu`；`--batch-size` 只改变推理批量，不改变拟合/评价样本。
支持 KGLD（及相同健康分支的现有 VAE/cVAE）、OCALD、Proposal-C，不支持 Residual。
KGLD 提取 HIHead 实际使用的完整 mu_d；OCALD 复用有效前缀适配后取完整 mu_I；Proposal-C 取1维Z。
全部 eval/no_grad 并冻结参数，不采样、不做 PCA、不微调。当前均每flight一次输出，重复键直接报错。
未来模型可向 `utils/linear_probe.py` 的键表接口提供已按 cycle 平均的完整慢状态，不新增空模型实现。

`representation` 与公共最终标量 `hi` 分别拟合独立的 sklearn `StandardScaler → LinearRegression(fit_intercept=True)`。
只在表征训练设备拟合新增 scaler/回归器；验证设备只作诊断，测试指标按全部cycle等权合并计算。
沿用已定的 X/W/截断/padding 和公共最终HI的全dev Min–Max；后者不是新增探测器的 scaler。
参考HI复用 `(unit,cycle)` 标签关联；缺cycle时沿用已验证行序推定，输出记录其无法独立验证语义顺序的限制。
训练划分优先核对 checkpoint 的 train/val units，或依据已记录 split_seed 和训练 val_split 重建；
信息缺失时必须提供 `--split-config` YAML，且文件仅含整数列表 `fit_units`、`val_units`，不得与已记录划分冲突。
不能按测试结果选择划分或checkpoint。当前DS03既有三个运行均核对为：fit=[1,2,3,4,5,7,8]，
val=[6,9]，test=[10,11,12,13,14,15]，分别516/147/438个cycle；该集合不是未知旧checkpoint的默认值。

输出 `probe_metrics.json`、`probe_predictions.csv`、`probe_features.csv`、`probe_parameters.json`、`config.yaml`。
分别报告探测后representation/hi的 pooled R²、MAE、RMSE及逐设备诊断，并单列未探测hi的相同指标；
记录特征维数、回归秩/秩亏/欠定、数值scaler/系数和fit键。常数目标或不足2个样本的R²为null并注明原因。
预测不做Sigmoid、输出Min–Max或裁剪。Proposal-C的Z与最终HI仅有仿射差别，两类探测可重复，不去重或夸大差别。
原 metrics.json/checkpoint 不改写。

方法参考 H-CDE commit `b5cfa1e069bd824ad65fc0ad427dd0e828ebf936`：
`src/train/task.py::get_latent_space` 的线性回归使用完整末端慢状态，PCA仅用于分析；
`src/model/hcde.py` 返回 d_all，`src/train/train.py::MappingTrainer.compute_loss` 将其记录为latent。
本项目只借鉴冻结表征→训练设备拟合→测试评价，改用DS03参考HI，不迁入H-CDE模型、DS01目标或其数据协议。

既有 W-summary 配对实验命令：

```bash
python -m experiments.run_w_summary_ablation \
  --output-dir runs/w_summary_pair_new --device cpu --threads 2
```

该命令会执行两次真实训练，仍使用现有两份 seed-42 YAML；应在明确安排实验后运行。
本轮只在隔离的短合成配置上核验调度，没有重跑正式配对训练。

每次配对实验保存 13 个文件：

```text
runs/<experiment_name>/
    comparison.csv
    comparison_curves_dev.png
    comparison_curves_test.png
    true/
        config.yaml
        best.pt
        history.csv
        metrics.json
        scores.csv
    false/
        config.yaml
        best.pt
        history.csv
        metrics.json
        scores.csv
```

`comparison.csv` 包含两组 dev/test 七项指标和 false 减 true 的差值；
两张图按发动机展示两组 HI 与参考 HI。每组不再单独生成曲线。
共同完成 epochs 的配对审计结果保存在各组 `config.yaml` 的 `run_metadata.paired_audit` 中。

## 批量线性探测对比

新增入口显式接收运行目录，支持完整 KGLD、三个条件先验/FiLM 消融、
三个 HSIC/pairwise/Spearman 损失消融及 Proposal-C；OCALD、Residual-A/B 记录为跳过。
不训练或微调网络，不改写 checkpoint 和原运行结果。

```bash
python -m experiments.run_linear_probe_comparison \
  --run-dirs runs/kgld \
    runs/knowledge_ablation_seed42/no_prior \
    runs/knowledge_ablation_seed42/no_film \
    runs/knowledge_ablation_seed42/no_prior_no_film \
    runs/loss_ablation_seed42/no_hsic \
    runs/loss_ablation_seed42/no_pairwise \
    runs/loss_ablation_seed42/no_spearman \
    runs/proposal_cnn_c \
  --output-dir runs/linear_probe_comparison_seed42 --device cuda --dry-run
```

正式批量探测去掉 `--dry-run`，只列出实际存在、已训练完成的运行目录；输出根目录必须不存在。
dry-run 只读取配置、checkpoint 及已有预处理缓存的设备键，不构建模型、不推理、不拟合回归器，
也不创建缓存或结果。划分调用现有 `resolve_probe_split()` 校验；缺少可靠划分或缓存出处时记录失败，
不根据运行名称、配置 seed 或测试结果猜测。单个运行失败后继续，最终输出成功/跳过/失败清单。

正常运行复用 `evaluate_linear_probe.run()`；结构消融局部绑定专用 checkpoint 加载器，
保留原探测流程，不修改现有模块全局函数。KGLD 所有变体使用完整 mu_d，Proposal-C 使用标量 Z。
沿用训练设备专属回归拟合、验证诊断、test cycle 等权合并、标签关联和公共 dev 全局 HI 缩放；
不对探测预测额外缩放、平滑或裁剪。Proposal-C 的 representation/HI 可能仅有仿射差别，两组结果都保留，
不视为独立收益。

每组输出到 `probes/<序号>_<运行目录名>/`，同名源目录也不会冲突；
保存 `config.yaml`、`probe_metrics.json`、`probe_predictions.csv`、`probe_features.csv`、
`probe_parameters.json`，补充来源、变体、实际 seed、维数和 fit/val/test 设备。
实际 seed 未记录时保留 null，不用 configured seed 替代。
根目录生成 `linear_probe_comparison.csv`、`linear_probe_comparison.md` 和 `comparison_summary.json`；
每个成功运行独立一行，不合并种子或配置，缺失指标不填零。原公共 HI 指标不变。

合成 checkpoint 的轻量检查（不训练网络）：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_linear_probe_comparison.py -v
```

## Knowledge ablation 独立线性探测

独立文件名为 `experiments/run_knowledge_ablation_linear_prob.py`（按此拼写），
仅比较完整 KGLD 和三个知识机制消融，复用已有严格加载及单次线性探测流程，不训练网络。

```bash
python -m experiments.run_knowledge_ablation_linear_prob \
  --run-dirs runs/kgld \
    runs/knowledge_ablation/no_prior \
    runs/knowledge_ablation/no_film \
    runs/knowledge_ablation/no_prior_no_film \
  --output-dir runs/knowledge_ablation_linear_prob --device cuda --dry-run
```

正式运行去掉 `--dry-run`；输出目录必须不存在。dry-run 只核对 checkpoint 元数据、已有缓存设备键、
参考 HI 键和值，不运行推理、不创建结果或缓存。
各组仍提取完整 mu_d，并仅在各自训练设备上拟合两类探测器；每组五类探测产物保存到独立 `probes/` 子目录。
根目录单独生成 **`knowledge_ablation_linear_prob.md`** 和 **`knowledge_ablation_linear_prob.csv`**，
两者共用汇总行；CSV 保留完整数值精度，未知 seed 和缺失指标留空。行名为 `KGLD (full)`、
`w/o conditional prior`、`w/o FiLM`、`w/o conditional prior and FiLM`。
报告含运行来源、实际 seed、特征维数、两类探测的 test R²/MAE/RMSE、checkpoint SHA256、
fit/val/test 设备，以及参考 HI 的路径/键值摘要和行序推定限制。
设备划分或参考 HI 不同会明确标记，不强制改为相同划分，不合并配置或种子。
实际 seed 未记录时保持未知，不将配置 seed 当作已验证的实际 seed。
本入口不会改写原批量探测脚本及其报告。

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_knowledge_ablation_linear_prob.py -v
```

## KGLD 知识机制消融

独立入口只读取启动时的 `config/kgld.yaml`，依次运行 `no_prior`、`no_film`、
`no_prior_no_film`，不重跑完整 KGLD，也不读取历史 VAE/cVAE 或完整模型运行配置。

```bash
python -m experiments.run_knowledge_ablation \
  --output-dir runs/knowledge_ablation_seed42 --seed 42 --device cuda --dry-run
python -m experiments.run_knowledge_ablation \
  --output-dir runs/knowledge_ablation_seed42 --seed 42 --device cuda
```

`--dry-run` 列出三组变体、配置差异和目录，不加载数据或模型、不创建结果。
正式运行时根目录必须不存在；每组重置模型随机种子，设备划分和分层采样仍用基准的 seed 42。
数据、模型维度、W 解码路径、训练预算/早停/阶段安排及全部损失参数保持当前 YAML 设置，
使用 `train_model()` 的原训练流程和最小验证重构损失选模规则。

`no_prior` 仅替换为两块标准正态先验，保留 FiLM 和原 KL 权重；`no_film` 仅替换工况后验分支，
复制原 TCN/head 权重并完全忽略 u，保留条件先验；第三组同时替换两者。
各组先正常初始化 KGLD，再替换模块，保留共同参数及初始化后的随机状态。

每个子目录保存实际完整 `config.yaml`、`best.pt`、`history.csv`、`scores.csv`、
`metrics.json` 和 dev/test 曲线；根目录保存六列 test 指标的 `comparison.csv`、`comparison.md`。
checkpoint 记录 `kgld_knowledge_ablation` 类型及变体，通过新增脚本的
`load_ablation_checkpoint()` 重建，训练后由 `evaluate_ablation()` 调用公共预测、标签关联、
dev Min–Max 和指标函数评价。公共 `evaluate.py` 和批量重评入口会拒绝将这些权重当完整 KGLD 加载。

轻量检查命令（合成输入，训练接口使用替身，仅保存初始化权重，无优化器更新）：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_knowledge_ablation.py -v
```

## KGLD 损失消融

独立入口读取当前 `config/kgld.yaml`，首先严格核对基准损失：`beta_d=0.1`、`beta_c=0.25`、
`lambda_hsic=1.0`、`lambda_mono=10`、`lambda_spearman=0.5`、`ranking_margin=0.01`、
`ranking_delta_t=3`、`spearman_tau=0.1`；不符则报错，不自动覆盖。

```bash
python -m experiments.run_loss_ablation \
  --output-dir runs/loss_ablation_seed42 --seed 42 --device cuda --dry-run
python -m experiments.run_loss_ablation \
  --output-dir runs/loss_ablation_seed42 --seed 42 --device cuda
```

根目录必须不存在。依次从头训练 `no_hsic`、`no_pairwise`、`no_spearman`，
每组仅将对应的 `lambda_hsic`、`lambda_mono`、`lambda_spearman` 设为 0。
其他参数、阶段和选模保持基准不变，不重跑 full。三组使用相同初始化 seed；
设备划分和分层采样保持现有 seed 42。`--dry-run` 只显示损失/运行信息差异，不创建结果目录。

每个子目录保存实际配置、带变体/seed/完整配置的 `best.pt`、`history.csv` 和公共评估产物。
根目录生成 `loss_ablation.csv`、`loss_ablation.md`，按每组 test 总体值列出
`Variant | MAE↓ | RMSE↓ | Mon↑ | Tren↑ | FC↑ | MI↑`；缺失指标保持为空，不填零。
full 的已有结果由后续自行合并。

各组 `checkpoint_report.json` 及根目录 `checkpoint_reports.json` 记录最佳 checkpoint 的阶段、
已训练轮数、是否曾进入时间约束阶段，以及最佳 checkpoint 是否处于该阶段。
损失函数使用零起始 epoch，checkpoint 保存已完成轮数；默认阶段边界 10/25 下，
最佳 checkpoint 的 `epoch >= 26` 才属于时间约束阶段。该报告不改变最小验证重构损失选模。

轻量验证命令如下；其中实际 CPU 训练仅使用临时合成数据，每组两轮，维度/预算缩小仅作用于测试配置，
不修改正式 YAML，不据此报告正式性能：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_loss_ablation.py -v
```

## 二维退化潜在空间与 HI 投影

先确认论文对应的运行及 checkpoint；`runs/kgld` 和 `runs/kgld_tuning` 是不同运行，
不能根据测试表现或目录时间选择。本次已由用户确认使用 `runs/kgld/best.pt`（epoch 198）：

```bash
python -m experiments.visualize_kgld_latent \
  --config runs/kgld/config.yaml --checkpoint runs/kgld/best.pt \
  --output-dir runs/kgld_latent_visualization --device cpu --batch-size 32 --threads 2
```

输出目录必须不存在。入口复用公共缓存、确定性表征提取、原始分数推理及参考 HI 键关联，
仅支持二维退化后验均值，不训练、不采样、不做 PCA 或监督线性探测。
提取全部 test flight，Fc 从各样本 `u` 末三位 one-hot 恢复；未核实航程语义前标为 Fc 1/2/3。
两幅图使用相同原始坐标及范围，分别按参考 HI 和 Fc 着色。箭头使用训练所得
`model.hi_head.linear.weight`，指向 HI 增大方向，不表示退化时间方向。

保存 `latent_points.csv`、`projection.json`、300 dpi PNG、矢量 PDF 和简短 `README.md`。
元数据包括实际配置、checkpoint SHA256、数据/参考标签来源、投影误差及同 checkpoint
历史分数对照；不会覆盖原结果。类别重叠本身不能证明完全解耦。

上述命令已在真实 DS03 上执行：438 个 test flight、发动机 10–15，Fc 1/2/3 分别为
169/67/202 个；`mu_d @ a` 与公共原始分数最大误差 `5.94e-8`。
与同 checkpoint 的既有 CPU 重评 `runs/main_results/evaluations/kgld/scores.csv` 完全一致。
原 CUDA 分数的最大差为 `1.67e-4`，超出严格对照容差，已在输出元数据中显式记录；
当前环境无 CUDA，未进一步隔离差异原因，不声称跨设备逐位复现。
PNG 已打开检查：参考 HI 沿点云呈明显梯度，Fc 类别存在重叠；不据此宣称完全解耦。

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_kgld_latent_visualization.py -v
```

## 退化与工况表示的 Flight category 对照

沿用上述已确认的论文 checkpoint，左图直接使用二维 `mu_d`，右图仅对八维 `mu_c`
做 PCA。PCA 在全部 dev flight 的原始后验均值上拟合，固定 `svd_solver="full"`，
不逐维标准化、不使用类别标签，test 仅 transform。两图均保留全部测试样本，使用相同
Fc 配色及各自等比例坐标轴；不添加 HI 箭头，不强制共享数值范围。

```bash
python -m experiments.visualize_latent_flight_category \
  --config runs/kgld/config.yaml --checkpoint runs/kgld/best.pt \
  --output-dir runs/kgld_latent_flight_category --device cpu --batch-size 32 --threads 2
```

输出目录必须不存在。保存 `latent_features.csv`（完整 dev/test 均值与键）、
`test_projection.csv`、`pca_parameters.json`（含 dev 拟合键和数值参数）、`metadata.json`、
300 dpi PNG、矢量 PDF 及简短说明；旧可视化结果原地保留。
Fc 从 `u` 的 one-hot 恢复并按 split/unit/cycle 关联，语义未核实前仍使用 Fc 1/2/3。
入口按 checkpoint SHA256 查找上次潜在空间结果，核对测试键和 `mu_d`；没有匹配产物时明确记录。

工况分支已通过 FiLM 接收包含 Fc 的条件输入，因此类别结构可支持其编码了航程类别信息，
不能单独证明完全解耦或工况识别的泛化能力。二维 PCA 未保留的信息也须纳入解释。

真实 DS03 运行使用 663 个 dev flight 拟合 PCA、438 个 test flight 绘图，测试 Fc 1/2/3
分别为 169/67/202 个。PC1 解释 35.62%、PC2 解释 31.83%，累计 67.44%；
新 `mu_d` 与上次同 checkpoint 产物完全一致。PNG 已打开检查：退化表示类别广泛重叠，
工况 PCA 中类别位置差异更明显，但仍有重叠，不能描述为完全分离的三团。

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_latent_flight_category.py -v
```

## 按马赫数和高度着色的连续工况对照

直接复用上一实验导出的完整后验均值、测试坐标和 dev 拟合 PCA，不重新推理或拟合：

```bash
python -m experiments.visualize_continuous_conditions \
  --latent-dir runs/kgld_latent_flight_category \
  --output-dir runs/kgld_continuous_conditions
```

输出目录必须不存在；缺少上一步导出时，先执行上一节命令，不以模拟表示替代。
新增一张 2×2 图：第一行原始二维 `mu_d`，第二行相同测试 flight 的 `mu_c` PCA，
两列分别按飞行平均马赫数和平均高度着色。同列共用 viridis 色条及完整测试范围，
同一行共享坐标，不裁剪、不筛样、不添加 HI 箭头。

通道映射读取原始 HDF 的 `W_var`，当前为 `[alt, Mach, TRA, T2]`。
工况均值读取缓存 `u` 前四维，来自每条 flight 下采样后、截断与补零前的完整有效序列。
马赫数使用缓存 dev 全局统计量反标准化；高度物理单位未获现有文件证实，
因此明确绘制 **dev 标准化均值**，色条标注 `Mean altitude (dev z-score)`，不猜测米或英尺。

保存 `plot_data.csv`、来源/计算公式/单位/PCA 与色条范围的 `metadata.json`、
300 dpi PNG、矢量 PDF 和简短说明。颜色梯度仅说明关联，不能单独证明完全解耦。

真实绘图包含 438 个测试 flight，复用的 PCA 解释方差仍为 35.62%/31.83%（累计 67.44%）。
全部原始 flight 的工况均值与缓存 `u` 精确一致，包括 63 条长度超过固定 `L_max` 的 flight；
马赫数反标准化相对直接原始均值的最大差约 `2.85e-7`。图片已打开并修正标签布局。
两类表示均出现区域性颜色变化及局部混合，不据此断言哪一种梯度普遍更清晰。

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_continuous_conditions.py -v
```

## 批量重评与论文主结果表

先列出待评估运行（不加载 checkpoint、不推理、不创建输出目录）：

```bash
python -m experiments.reevaluate_runs --runs-dir runs \
  --output-dir runs/main_results_20261020 --device cuda --dry-run
```

确认列表后，去掉 `--dry-run` 正式重评；CPU 可改用 `--device cpu`：

```bash
python -m experiments.reevaluate_runs --runs-dir runs \
  --output-dir runs/main_results_20261020 --device cuda
```

输出目录必须不存在。递归扫描时排除所有 `history/` 子树，不跟随目录符号链接。
含 `config.yaml`、`best.pt`、`history.csv`、`metrics.json` 或 `scores.csv` 的目录，
以及非根空叶目录作为候选；缺少 `config.yaml` 或 `best.pt` 记为跳过，
无运行产物的中间分组目录不计为运行。每个完整运行调用公共 `evaluate.py`，
沿用 checkpoint 实际配置及校验，在新目录的 `evaluations/<原相对运行路径>/` 保存
`metrics.json`、`scores.csv` 和 dev/test 曲线；不会改写原运行产物。

根目录生成 `main_results.csv`、`main_results.md` 和 `summary.json`。
主表每个成功运行一行，列为方法、运行名称、MAE↓、RMSE↓、Mon↑、Tren↑、FC↑、MI↑；
方法按公共评估器返回的实际配置区分，运行名称保留变体，不合并均值或标准差。
仅提取 test 总体指标，RMSE 直接采用公共评估中按全部 cycle 合并的结果，不平均逐设备 RMSE。
缺失指标在 CSV 留空、Markdown 显示 `—`；MAPE、dev、逐设备指标及完整配置仍在各运行详细结果中。
单个运行失败后继续，汇总记录成功、跳过、失败及原因/错误 traceback。

## 数据与评估协议

输入为 N-CMAPSS HDF5 的 `A/W/X_s/Y`，按 unit/cycle 分组并覆盖完整生命周期。
保持逐 cycle 下采样、P90 长度、逐 unit 传感器标准化、dead-sensor 置零、
头部 padding 和保留 tail 的截断。W 全局归一化后分别构造完整 cycle 的 u 和模型 W_seq；
不改变先验、FiLM 或 decoder 的条件路径。未见 test unit 的传感器统计按现有实现单独计算。

协议 `unsupervisedhi-ncmapss-v1` 按真实 flight 键关联分数：KGLD 保持 posterior mean 的线性 raw HI，
OCALD 使用 -score(mu_I)，Proposal-C 使用 Z。
仅在全部 dev 分数上拟合全局 MinMaxScaler，再 transform dev/test；
不逐机缩放、不 EWMA、不裁剪。test HI 可以超出 `[0, 1]`。
七项指标为 MAE、RMSE、MAPE、Monotonicity、Trendability、FC、MI；MI random_state=42。
指标含义、计算公式与汇总规则见 [健康指数评价指标](docs/HI_METRICS.md)，另列独立线性探测指标。
无 HI 参考时 MAE/RMSE/MAPE 为 null，RUL 仍用于 MI；FC 与旧 Prognosability 不能直接混用。
参考 HI 若无 cycle 列，按现有严格行序检查关联，其语义行序的独立核实限制保留。

## 运行产物与复现范围

训练并自动评估后保存 `config.yaml`、`best.pt`、`history.csv`、`metrics.json` 和 `scores.csv`。
普通训练和独立评估仍生成 dev/test 曲线；独立评估只保存评估结果及曲线。
归一化参数保存在 `metrics.json` 的 `metadata.scaling`，完整评估出处和配置保存在 `metadata`，
不再额外生成 `scaler.json` 或 `evaluation_config.yaml`。已有运行文件保持原样。
KGLD snapshot 记录实际维度、路径、源配置、已知 seed、初始权重/RNG hash、设备和线程。
普通入口保留原未固定初始化 seed 的行为：实际 seed 未指定时记录 null，
YAML 中的 configured_seed 单独记录；配对脚本明确设置 seed=42。
hash 是审计依据，不等同于保存完整初始化/RNG 状态。

checkpoint 保存 state_dict 字典。新文件保留已有参数 keys 和保存字段；
旧 checkpoint 原地保留。训练加载函数不完整恢复 scheduler、sampler 和 RNG，
没有精确续训 CLI。decoder 仍固定输出 14 个传感器，不支持任意 S。

## 验证与文档

旧 tests/ 已在此前清理中删除；本次按迁移验证要求新增永久 unittest 测试：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

此前迁移的 8 项测试通过，包含四份配置的 CPU CLI smoke、保存重载、冻结/联合梯度、
卷积/Flatten/损失及 Adam 独立公式对照、完整 dev 缩放与六指标/CSV 检查。
smoke 在测试创建的合成数据上自动加 --smoke 执行两个入口；临时配置、数据、权重和输出由 TemporaryDirectory 清理。
另对照固定 kgld commit：三份 OCALD 的初始化、随机前向、损失、参数梯度及联合配对序列完全一致；
旧 KGLD checkpoint 的真实 DS03 1101 个周期分数、完整 metrics 和曲线与修改前一致。
没有 TensorFlow 环境或源 CNN 权重，未做跨框架逐位复现，也未执行完整 50/100 轮训练或 GPU 验证。

W-summary 配对脚本在运行时检查配置、共享初始权重和共同训练 epochs 的 batch/RNG/学习率摘要，
摘要保存在 history 和配置中。不再提供独立 verify 命令，也不依赖旧 Git revision 或历史 checkpoint。
2026-10-18 Residual 接入验证：10 项 unittest 通过，包含六份配置的独立 CPU CLI smoke。
合成数据含每设备24个周期；Residual 每次仅更新7条健康段样本，但 PCA/scaler 使用全部144条 dev，
test 为48条；验证更新键、输入路径、独立卷积/MAE及梯度、展平、sklearn PCA/方向、保存重载及六指标。
smoke 命令由 `tests/test_experiment_smoke.py` 使用上述训练命令加 `--device cpu --smoke` 执行；
旧 KGLD DS03 checkpoint 的1101条 flight 在本次修改前后重评，scores、metrics、两张曲线逐字节一致。
测试临时数据/配置/权重/曲线自动清理。未运行完整20轮或 TensorFlow/GPU 对照。

2026-10-18 短窗口验证：13 项 unittest 通过，覆盖八份配置 CPU CLI smoke；
归一化提取前后固定张量完全一致，源 sequence_generator 的2850个合成窗口 X/W/键逐元素一致。
新增测试核对窗口边界/数量、未截断序列、统计复用、短 flight 报错、PCA/10窗口方向/flight均值顺序、
推理 batch 变化容差、完整键覆盖、重载和 dev-only 缩放。旧 KGLD、flight A/B 的真实 DS03
checkpoint 重评：scores、metrics、两张曲线分别与修改前逐字节一致。

独立真实 DS03 smoke（均已清理临时输出）：

```bash
OPENBLAS_NUM_THREADS=8 python -m experiments.train_residual --config config/residual_a_short_window.yaml \
  --run-dir /tmp/residual_window_source/real_a --device cpu --threads 8 --smoke
OPENBLAS_NUM_THREADS=8 python -m experiments.train_residual --config config/residual_b_short_window.yaml \
  --run-dir /tmp/residual_window_source/real_b --device cpu --threads 8 --smoke
```

A/B 均使用50/stride1，候选健康段为140个 train flight、112402个窗口；各只更新1步/512个窗口。
PCA 使用完整663个 dev flight的524907个窗口，test为438个flight的403868个窗口；
公共 scaler 在663条聚合后的 dev 分数上拟合，最终 CSV 为1101条 flight。
独立重载参数、六指标重算和键覆盖均通过；真实样本固定权重下 batch127/512 的最终分数一致。
保留正式变长缓存和永久测试，清理本次临时配置、权重、残差矩阵、图及下载参考文件。
未运行完整20轮、TensorFlow/GPU对照，不将 smoke 或输入组织变化解释为性能改善。

线性探测验证：18项unittest通过，新增5项覆盖完整均值/Z、无随机采样/权重更新、键覆盖、
仅fit设备拟合、线性回归数值参数恢复、缺失划分/秩亏/未定义R²，以及三个随机权重合成CLI。
实际DS03三个已有checkpoint还完成独立CPU探测检查（8线程、临时目录，未更新网络）：
mu_d/μ_I/Z维数分别2/8/1，fit/val/test均为516/147/438；两类probe秩分别2/1、8/1、1/1。
三模型公共评价修改前后，1101条flight的scores、完整metrics及两张曲线逐字节一致。
命令为上述三个probe命令，验证时改用 `--device cpu --threads 8` 和独立 `/tmp/linear_probe_checks/` 输出；
合成及真实检查的临时数据/权重/输出均清理。未运行网络训练或GPU验证，不将带标签的探测分数当无监督主指标。

历史审计报告保留原有记录；新实验不生成额外的历史对照或验证目录。

CUDA 配对实验须在启动 Python 前设置确定性计算环境变量：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m experiments.run_w_summary_ablation \
  --output-dir runs/w_summary_pair_cuda_new --device cuda --threads 2
```

重构证据见 [第一步](docs/history/REFACTOR_STEP1.md)、[第二步](docs/history/REFACTOR_STEP2.md)、
[第三步](docs/history/REFACTOR_STEP3.md)、[第四步](docs/history/REFACTOR_STEP4.md)、[第五步](docs/history/REFACTOR_STEP5.md)。
当前代码说明见 [CODE_WALKTHROUGH.md](docs/CODE_WALKTHROUGH.md)。
原状态和走读保留在 [docs/history/](docs/history/)，设计、旧审计和评估验证文档保留历史结论。
CPU 上模型/损失/梯度/采样对照通过；DS03 旧 checkpoint 的 1101 个 flight 重评一致。
本轮真实数据仅做 Residual 单步 smoke；没有完整预算训练、GPU 验证或真实 DS02 实验，不称为完整实验复现。

2026-10-19 模型更名为 KGLD：文件/类/入口/线性探测及当前文档引用已同步。
主配置为 config/kgld.yaml，旧配置和state_dict权重不迁移；旧模型类型oc_ivae仅作为名称别名解析。
原18项测试与新增旧类型加载测试通过；固定输入/随机状态的初始化、keys、前向、三阶段损失和梯度完全一致，
历史DS03 checkpoint的1101条flight重评分数、完整metrics和曲线逐字节一致。未执行完整训练或GPU验证。
