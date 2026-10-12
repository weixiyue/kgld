# OCALD

基于 PyTorch 的 N-CMAPSS DS03 flight 级健康指标（HI）学习。输入完整 flight 的传感器序列，先学习潜变量表示，再利用同设备 flight 的时间顺序训练 HI 头。

当前支持原始 dev/test 划分、两阶段训练、smoke 验证、checkpoint 独立加载、raw/scaled HI 评价与曲线导出。未实现其他数据集、下游 RUL 预测、联合微调或基线/消融实验。HI ground truth 和 RUL 仅用于评价，不参与训练、选模或早停。结构不保证工况解耦，也不提供物理尺度标定。

## 1. 网络与训练

| 模块 | 实际结构 | 输入与输出 |
|---|---|---|
| SensorEncoder | 单层单向 GRU，14→64；四个 Linear，分别为 64→8 | X/mask → mu_I、logvar_I、mu_C、logvar_C |
| ConditionalPrior | 单层单向 GRU，4→32；两个 Linear，分别为 32→8 | W/mask → z_C 条件先验的均值和 log variance |
| Decoder | 共享时间条件 MLP：17→64→64→14；隐藏层 ReLU，输出线性 | [z_I, z_C, 时间位置] → X 重构 |
| HIHead | MLP：8→16→1；隐藏层 ReLU | mu_I → r；hi_raw=sigmoid(-r) |

GRU 按真实长度打包，取最后有效隐藏状态；padding 不参与重构损失。时间位置在每条 flight 内归一化为 0..1，单点序列为 0。网络使用 PyTorch 默认初始化，无额外 dropout 或 log variance 裁剪。

**信息路径：**X 生成两组对角高斯后验；W 仅调制 z_C 的对角高斯先验。z_I 的先验为标准正态。Decoder 不直接读取 W，HI 推理只读取 X/mask，使用确定性的 mu_I，不读取 W、cycle 或评价标签。

- **Stage 1（representation）：**训练编码器、条件先验和解码器，后验采用重参数化采样。目标为重构项加两个解析 KL：重构先对每条 flight 的有效时间步及传感器求和，再与 KL 相加，最后按 flight 平均。HI 头不训练。
- **Stage 2（hi）：**加载 Stage 1 权重，将表示模块设为 eval 并冻结；提取全部 dev flight 的 mu_I，仅训练 HI 头。排序项为 `softplus((r_early-r_late)/kappa)`，加包含偏置的显式头参数 L2。同设备周期排序与样本对构造在同一设备上执行，cycle 不作为网络输入。

两阶段均保存最后一轮权重，不使用早停、学习率调度或测试指标选模。

## 2. 环境

在项目根目录执行；以下命令创建独立环境并安装当前依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` 未锁定版本。实际验证过的 Linux 环境如下，其他组合未验证：

| 组件 | 已验证版本 |
|---|---|
| Python / PyTorch | 3.11.15 / 2.5.1+cu121 |
| NumPy / pandas / h5py | 2.4.6 / 3.0.3 / 3.16.0 |
| SciPy / scikit-learn | 1.17.1 / 1.9.1 |
| PyYAML / matplotlib | 6.0.3 / 3.10.9 |

CPU smoke 和 RTX 3090 单卡 CUDA 正式训练均曾执行。训练和评价通过 `torch.cuda.is_available()` 自动选择 CUDA，否则 CPU；没有 CLI 设备覆盖参数。可检查当前进程是否看得到 GPU：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

CPU 训练可在命令前加 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`，此前 CPU smoke 使用了这两个环境变量。matplotlib 使用 Agg 后端，输出 PNG，不需要桌面环境。

## 3. 数据准备

1. 从 NASA 的 N-CMAPSS 数据集准备 DS03 原始 HDF5，当前配置对应 `N-CMAPSS_DS03-012.h5`。数据集说明见 [NASA NTRS](https://ntrs.nasa.gov/citations/20205001125)。本项目不提供自动下载器。
2. 从参考仓库的 [N-CMAPSS_HI_data](https://github.com/KBaja/UnsupervisedHI/tree/157ebfae7baeee1e0cd52180f8e58ead3dcb448e/N-CMAPSS_HI_data) 下载两个 DS03 cycle HI 文件；它们是参考 HI ground truth，不是模型预测。
3. 将文件放入下面的相对目录，或在自己的 YAML 配置中设置三个数据路径：

```text
data/raw/
  N-CMAPSS_DS03-012.h5
  N-CMAPSS_DS03_cycle_dev.h5
  N-CMAPSS_DS03_cycle_test.h5
```

原始文件需要 `X_s_var`、`W_var`、`A_var` 及 dev/test 的 `X_s`、`W`、`A` 数组；A 中需要 unit/cycle。评价另读取 `Y_dev/Y_test` 的 RUL，要求 RUL 在每条 flight 内恒定、有限且非负。HI 文件按已核对的 pandas fixed HDF 格式读取 `df` 中的 unit/HI，不需要 PyTables。

当前 DS03 的 dev 为 unit 1–9、663 条 flight，test 为 unit 10–15、438 条 flight。X_s/W 分别为 14/4 维，下采样后长度为 312–2029。训练预处理不读取 HI/RUL。

**预处理口径：**X/W 各列的 [0,1] Min–Max 参数由下采样前全部原始 dev 行拟合；test 复用训练参数，不裁剪。每个 unit 连续每 10 点取 1 点，X/W 共用采样索引，不在 flight 边界重启采样；随后按 unit/cycle 分组，保留完整 flight。mask 由真实长度生成，不根据数值是否为零判断。

HI 表没有显式 cycle。读取时检查设备块、行索引、每台设备行数及原始周期为 1..N，再按设备内存储行顺序生成 cycle；这是参考文件的行顺序约定，不能独立证明标签周期正确。预测、HI 和 RUL 随后按 `(unit, cycle)` 一对一关联，缺失或重复键会报错。

## 4. 配置

所有入口都要求显式 `--config`，没有默认配置文件。下面数值来自当前 `config.yaml`；模型、数据口径及训练配置会被代码严格校验，不能仅改 YAML 任意调参。

| 配置项 | 当前值与含义 |
|---|---|
| `data.raw_path / hi_dev_path / hi_test_path` | 上述三个 `data/raw/` 文件 |
| `data.processed_path` | `data/processed/ds03.pt`，预处理输出 |
| `sensor_field / condition_field` | `X_s / W`；维度 `14 / 4` |
| `downsample_stride / downsample_scope / sample_scope` | `10 / unit / full_flight` |
| `normalization / normalization_fit / normalization_clip` | `minmax_0_1 / raw_dev_before_downsampling / false` |
| `model` | X/W GRU 隐藏维度 `64/32`，`gru_layers=1`、`bidirectional=false`；潜变量 `8/8`，Decoder 隐藏层 `[64,64]`，HI 隐藏层 `16` |
| `training.seed / device / checkpoint_selection` | `42 / auto / last_epoch` |
| 两阶段 `optimizer / learning_rate / weight_decay` | `Adam / 0.001 / 0.0`，其余 Adam 参数为 PyTorch 默认值 |
| Stage 1 `batch_size / epochs` | `16` 条 flight / `100` epochs |
| Stage 1 `sigma_x_squared / kl_i_weight / kl_c_weight` | `1.0 / 1.0 / 1.0`；固定重构方差与两个 KL 系数 |
| Stage 1 `gradient_clip_norm` | `5.0`，全局梯度范数上限 |
| Stage 2 `batch_size_pairs / epochs` | `128` 对 / `100` epochs |
| Stage 2 `min_cycle_gap / kappa` | `5 / 1.0`；早晚周期差至少 5，排序温度为 1 |
| Stage 2 `l2_coefficient / include_bias_in_l2` | `0.0001 / true`；显式 L2 包含偏置，不使用 weight decay 替代 |
| Stage 2 `sampling / pairs_per_epoch` | `uniform_unit_then_uniform_valid_pair / N_train`；先均匀选有效设备，再均匀选该设备全部有效早晚对，允许重复；每轮抽训练 flight 数量的对 |
| `smoke` | Stage 1 最多 `32` 条 dev flight；两阶段各 `3` epochs；Stage 2 使用全部 dev 表示 |
| `output.root` | `outputs`，训练与数据查看图的输出根目录 |

## 5. 从预处理到评价

所有命令在项目根目录运行。每次训练打印实际 `run_dir`，并创建新的 UTC 时间戳目录；下文的路径变量需替换为对应命令打印的目录，不会自动寻找最近一次运行。

### 5.1 预处理与数据查看

```bash
python data.py --config config.yaml --prepare
python verify.py --task data --config config.yaml
python verify.py --task model --config config.yaml
python verify.py --task metrics --config config.yaml

python verify.py --task data-plot --config config.yaml --split dev --unit 1 --cycle 1 --parameters T24 T30 Nf alt
```

预处理写入 `data/processed/ds03.pt`，重复执行会覆盖它。`data` 验证重新检查全部采样元素和归一化，需要原始文件及 HI 表。`data-plot` 输出 `outputs/data_inspection/dev_unit_1_cycle_1.png`：所选 flight 的归一化参数曲线，以及 unit 1 的完整参考 HI。参数名可以选择 X_s/W 实际列名；未知或重复参数会报错，横轴原始行位置不表示秒。

### 5.2 Smoke：检查两阶段流程

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --config config.yaml --stage representation --smoke
REP_SMOKE="outputs/representation_smoke_实际时间戳"
python verify.py --task representation --config config.yaml --checkpoint "$REP_SMOKE/representation.pt"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --config config.yaml --stage hi --representation "$REP_SMOKE/representation.pt" --smoke
HI_SMOKE="outputs/hi_smoke_实际时间戳"
python verify.py --task hi --config config.yaml --checkpoint "$HI_SMOKE/ocald.pt"
python evaluate.py --config config.yaml --checkpoint "$HI_SMOKE/ocald.pt"
python verify.py --task evaluation --config config.yaml --run-dir "$HI_SMOKE"
```

Stage 1 用 seed=42 固定抽取最多 32 条完整 flight；归一化仍由完整 dev 拟合。Stage 2 smoke 提取全部 663 条 dev 表示，运行 3 epochs。smoke 指标只验证流程，输出与正式运行分开，不作为正式实验结果。

### 5.3 正式 Stage 1 → Stage 2 → 评价

```bash
python train.py --config config.yaml --stage representation
REP_RUN="outputs/representation_formal_实际时间戳"
python verify.py --task representation --config config.yaml --checkpoint "$REP_RUN/representation.pt"

python train.py --config config.yaml --stage hi --representation "$REP_RUN/representation.pt"
HI_RUN="outputs/hi_formal_实际时间戳"
python verify.py --task hi --config config.yaml --checkpoint "$HI_RUN/ocald.pt"

python evaluate.py --config config.yaml --checkpoint "$HI_RUN/ocald.pt"
python verify.py --task evaluation --config config.yaml --run-dir "$HI_RUN"
```

不加 `--smoke` 即分别运行 100 epochs。正式 Stage 2 拒绝 smoke 表示 checkpoint。训练没有恢复中断或从已有权重续训的 CLI。

`evaluate.py` 同时生成 dev/test 预测、两套指标和全部设备曲线；没有单独绘图参数或输出目录参数。输出固定写入 `ocald.pt` 所在目录，重复评价会覆盖同名 CSV/JSON/PNG。

## 6. 输出与评价口径

| 文件 | 所在位置与内容 |
|---|---|
| `ds03.pt` | `data/processed/`；dev/test flight、列名和训练归一化参数，无 HI/RUL 标签 |
| `config.yaml` | 每个训练目录中的配置快照 |
| `losses.csv` | 每个训练目录；Stage 1 记录 loss/reconstruction/kl_i/kl_c，Stage 2 记录 loss/order/l2；l2 为未乘系数值 |
| `representation.pt` | Stage 1 目录；表示模块权重、配置、训练归一化与阶段信息，不含受训 HI 头 |
| `ocald.pt` | Stage 2 目录；完整模型、配置、归一化、训练 HI Min–Max 与表示来源模式/轮数 |
| `predictions.csv` | HI checkpoint 目录；每条 flight 一行 |
| `metrics.json` | 同目录；dev/test 的 raw/scaled 整体与适用的逐设备指标、计数及训练缩放参数 |
| `hi_curves.png` | 同目录；每台设备完整 raw/scaled 曲线、参考 HI、设备编号和指标 |

预测表列：`split` 为 dev/test，`unit/cycle` 为唯一 flight 键；`hi_raw` 为原始 Sigmoid HI，`hi_scaled` 为训练预测缩放后的 HI，`hi_gt` 为参考 HI，`rul` 为仅供 MI 评价的原始 RUL。

`hi_scaled=(hi_raw-train_min)/(train_max-train_min)`，train_min/max 仅由全部 dev 的最终 raw 预测拟合，保存在 checkpoint 中。评价不重新拟合、不逐设备缩放、不裁剪或平滑，test scaled 可以超出 [0,1]。训练极差为零时 scaled 不可用，CSV 该列为空、JSON 标记不可用；`scale_hi` 直接调用会报错，不加 epsilon。

| 指标 | 实际汇总口径 |
|---|---|
| MAE / MAPE (%) | 按 flight 汇总误差；MAPE 排除 hi_gt=0，报告有效数量 |
| 单调性 M | 正与非正相邻差分数量差的绝对值除以差分数，零差分归非正；整体取设备均值 |
| 趋势性 | 逐设备计算 HI/cycle 的有符号 Spearman，整体先取均值再取绝对值；图中 rho 为设备有符号值 |
| 失效一致性 FC | `exp(-std(各设备末HI, ddof=0)/mean(首HI−末HI))`；仅整体，不报告逐设备 FC |
| MI | sklearn 默认 3 邻居估计 HI/RUL 互信息，seed=42；取 `1-exp(-MI)`，整体为设备均值 |

按参考实现，常数 HI 的 M 为 1，Spearman 的 NaN 记为 0；常数 HI/RUL 的 MI 直接交给 sklearn。参考未考虑的小样本、空 MAPE、零分母等暂不增加处理，可能产生错误或非有限结果；JSON 禁止写入非有限数。FC 超过 1 时保留原公式值并说明原因。常数序列的高单调性分数不表示学到了有效健康变化。

## 7. Checkpoint 独立加载

```python
import torch
from train import load_representation, load_ocald, scale_hi

representation, rep_checkpoint = load_representation(
    "outputs/representation_formal_实际时间戳/representation.pt", torch.device("cpu")
)
model, checkpoint = load_ocald(
    "outputs/hi_formal_实际时间戳/ocald.pt", torch.device("cpu")
)
# x: float32 [B,T,14]，按 checkpoint['normalization']['x'] 归一化。
# mask: bool [B,T]，有效前缀后接 padding；x/mask 与模型在同一设备。
with torch.no_grad():
    hi_raw = model.predict_hi(x, mask)
    hi_scaled = scale_hi(hi_raw, checkpoint["hi_minmax"])
```

两个加载函数均只需对应 checkpoint，返回 `(model, checkpoint)`，并设为 eval；`load_ocald` 还冻结表示模块。HI 推理不需要表示源文件、原始 HDF5、预处理文件或参考标签。评价 CLI 与此不同：它仍需原始文件、预处理数据及两份 HI 表，并要求传入配置与 checkpoint 完全一致，包括路径和 output.root。

## 8. 验证与常见问题

| `verify.py --task` | 必需的额外参数 | 检查范围 |
|---|---|---|
| `data` | 无 | 全量采样/归一化、mask、X/W 对齐和 HI 关联 |
| `data-plot` | `--split --unit --cycle --parameters` | 单 flight 参数与完整设备参考 HI 绘图 |
| `model` | 无 | 真实前向/反向、信息路径、padding、解析损失和排序方向 |
| `representation` | `--checkpoint` | Stage 1 更新、最后轮数、日志及加载一致性 |
| `hi` | `--checkpoint` | 表示冻结、头更新、样本对、推理及训练缩放 |
| `metrics` | 无 | 手算指标、参考行为、标签键关联和零尺度标记 |
| `evaluation` | `--run-dir` | 重新推理，对照 CSV、标签、指标和曲线，检查正式 100/100 epochs |

所有任务还需 `--config`。成功打印 `PASS` 并退出 0；参数缺失或检查失败返回非零退出码。训练产物验证还需要预处理数据及 checkpoint 同目录的 losses.csv；评价验证需要同目录 CSV/JSON/PNG。

- **配置不匹配：**更改数据设置后需重新 `--prepare`；Stage 2 检查表示 checkpoint 的模型/数据设置、列名和归一化；评价及 checkpoint 验证要求完整配置一致。
- **加载与评价跨设备：**不承诺 CPU/CUDA 逐位相同。评价验证对 CSV 与重新推理使用精确比较，重验已有 CSV 应使用导出时的设备环境。
- **文件格式或关联错误：**本项目只支持当前 DS03 字段及 HI fixed HDF 布局，不会自动选列或猜测标签对应关系。
- **GPU 不可见：**检查当前进程的 CUDA 可见性；受限环境可能回退 CPU。代码不会自动缩短 flight 或降低 batch。

当前已有真实数据、模型、指标、两阶段 CPU smoke，以及 CUDA 100/100 epochs 正式训练与导出验证记录。本次 README 重写仅重验 CLI、环境、模型、指标及 checkpoint 独立加载等轻量检查，没有重跑预处理、smoke 或正式训练，也没有重新导出评价结果。临时 smoke、数据查看图和未完成训练输出已清理，保留两阶段正式产物。

## 9. 文件与参考来源

实现平铺于根目录：`data.py` 数据、`model.py` 网络、`losses.py` 目标、`train.py` 两阶段训练及加载、`metrics.py` 指标、`evaluate.py` 导出，`verify.py` 为统一验证入口。数据、缓存、权重及 outputs 被 Git 忽略；项目仅使用本地 Git。

- [UnsupervisedHI 参考仓库](https://github.com/KBaja/UnsupervisedHI)：采样、参考 HI 与指标口径；对应论文为 Bajarunas 等，*Health index estimation through integration of general knowledge with unsupervised learning*（2024）。
- 核对版本 `157ebfae7baeee1e0cd52180f8e58ead3dcb448e`：[预处理 Notebook](https://github.com/KBaja/UnsupervisedHI/blob/157ebfae7baeee1e0cd52180f8e58ead3dcb448e/CMAPSS/01_data_preprocess.ipynb)、[采样函数](https://github.com/KBaja/UnsupervisedHI/blob/157ebfae7baeee1e0cd52180f8e58ead3dcb448e/CMAPSS/pre_process_utils.py)、[评价函数](https://github.com/KBaja/UnsupervisedHI/blob/157ebfae7baeee1e0cd52180f8e58ead3dcb448e/CMAPSS/eval_utils.py)。本项目独立实现 PyTorch OCALD，不声称复现参考仓库全部模型或实验。
- 开发约定见 [AGENTS.md](AGENTS.md)，设计对照见 [OCALD_Implementation_Plan.md](OCALD_Implementation_Plan.md)；使用说明以当前代码为准。

## 10. D1：内部表示尺度诊断

`diagnose.py` 的 `scale` 实验冻结同一正式 Stage 1 表示，提取全部 dev/test 的 mu_I，
分别重新训练 `zi_raw` 和 `zi_standardized` 两个独立 HI 头。标准化逐维使用全部 dev
的均值及总体标准差（ddof=0），零标准差的分母为 1 并记录维度，test 复用 dev 参数。
两头结构、初始权重、每轮配对及批次顺序相同，优化器独立，采样随机状态与初始化隔离。
训练沿用 Stage 2 配置，smoke 为两组各 3 epochs，正式为两组各 100 epochs，均保存最后一轮。
该变体仅用于诊断，不替换正式 OCALD；HI 标签只在两头完成训练后读取。

```bash
REP_RUN="outputs/representation_formal_实际时间戳"
python diagnose.py --config config.yaml --representation "$REP_RUN/representation.pt" --experiment scale --smoke
python diagnose.py --config config.yaml --representation "$REP_RUN/representation.pt" --experiment scale
```

每次创建新的 `outputs/diagnostics_<UTC时间戳>/`，正式实验重新初始化，不继承 smoke。
先保存 `latent_stats.csv`（dev/test 总体逐维统计及 dev 每台设备内部统计），然后训练。
输出 `comparison.csv`、`pair_schedule.pt`、`verification.json`、配置快照，以及两组各自的
`losses.csv`、`predictions.csv` 和 `head.pt`；头文件包含初始/最终参数及必要标准化参数。
两张图为 `training_comparison.png`（order 与加权 L2）和 `hi_comparison.png`（全部设备 raw HI）。

评价枚举每台设备全部周期差≥5的有效对，以退化分数 r 的 `r_late>r_early` 判断正确，
平局单独统计；`comparison.csv` 的 `unit=all` 行对 accuracy/tie_rate/order/有符号 Spearman
按设备等权平均。整体 HI 范围、总体标准差及 MAE 按全部 flight 计算；不做 HI Min–Max。
验证检查表示参数/缓冲区未变、头初始参数相同、配对批次一致、仅 dev 拟合及两组
重新加载后 dev/test 分数逐值一致；成功末行显示 `PASS: D1 ...`，失败返回非零退出码。
这里只支持 D1；其他表示比较、少量配对拟合或联合训练尚未实现。

本次 D1 使用 `outputs/representation_formal_20261010T102117772658Z/representation.pt`。
CUDA smoke（各 3 epochs）输出为 `outputs/diagnostics_20261012T055435561291Z/`，
正式（各 100 epochs）输出为 `outputs/diagnostics_20261012T060208463300Z/`，均通过上述验证。
另独立核对 CSV 中全部有效对计数、正确/平局计数、设备等权汇总及 loss 分项；
手算例区分设备等权与配对加权，两张图已查看。正式源 checkpoint 哈希未变。

| 数据 | 实验 | 设备等权排序正确率 | 穷举有效对的设备等权 order | 平均有符号 HI/cycle Spearman |
|---|---|---|---|---|
| dev | zi_raw | 53.75% | 0.688848 | −0.097772 |
| dev | zi_standardized | 58.26% | 0.674458 | −0.213557 |
| test | zi_raw | 51.25% | 0.694473 | −0.032071 |
| test | zi_standardized | 55.99% | 0.688696 | −0.148830 |

dev/test 分别穷举 21965/14429 对，两组平局均为 0。dev 逐维总体标准差为
0.015151..0.081751，各设备内部逐维标准差为 0.011808..0.093462，无零标准差维度。
标准化使 dev/test 正确率分别提高约 4.52/4.74 个百分点；9 台 dev 均改善，6 台 test 中 5 台改善。
这支持当前头对表示的利用受中心化/尺度处理影响，但排序能力仍弱；标准化也改变了相同 L2
在原始坐标中的有效约束，单次实验不能区分尺度、优化与正则作用，也不能认定退化信息已丢失。
这是重新训练的配对对照，不以旧 ocald.pt 代替 raw 对照。D1 已完成，等待用户验收。
