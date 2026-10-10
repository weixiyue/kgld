# OCALD

首版范围为 PyTorch、N-CMAPSS DS03、完整 flight 输入和 flight 级 HI。
执行依据是 [AGENTS.md](AGENTS.md) 与 [实现方案 v2](OCALD_Implementation_Plan.md)。
每个任务完成验证及本地 Git 提交后，等待用户明确验收；设计批准不代表跨任务启动授权。

## 当前任务记录

| 任务 | 状态 | 验证摘要 | 用户确认记录 |
|---|---|---|---|
| T0 初始化 | 已验收 | 依赖、路径及忽略规则通过；commit b98f60d | 用户回复“开始T1”，确认继续 |
| T1 数据管线 | 已验收 | 全量数据验证、data-plot 通过；commit 0c4b4aa | 用户回复“确认通过，开始 T2” |
| T2 模型与损失 | 已验收 | 真实前向/反向及机制验证通过；commit 40289d6 | 用户回复“开始 T3”，确认继续 |
| T3 表示学习 | 实现及 smoke 完成，待验收 | 32 条真实 flight、3 epochs；更新、日志、checkpoint 独立加载通过；正式训练未运行 | 用户已授权执行；待验收 |
| T4 排序学习 | 未开始 | 未验证冻结、采样或推理 | 设计与预算已批准；待 T3 验收 |
| T5 评价 | 未开始 | 未计算评价指标 | 待 T4 验收 |

T0 只创建本 README、requirements.txt、.gitignore 和基础 config.yaml，
并将用户提供的两份规则与方案文件纳入首次本地提交。Python 实现文件按后续任务创建。
T1 新增 data.py 与 verify.py 的数据验证入口，更新数据配置和本文记录。
T2 新增 model.py、losses.py，并扩展 verify.py 的模型机制检查。
T3 新增 train.py 的 Stage 1 入口与 checkpoint 加载，扩展 verify.py 的表示训练检查。

## 环境与 T0 验证

使用当前 Python 环境运行：

```bash
python -m pip install -r requirements.txt
python -c "import torch, numpy, pandas, h5py, scipy, sklearn, yaml; print(torch.__version__)"
python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"
python -c "from pathlib import Path; import yaml; c=yaml.safe_load(Path('config.yaml').read_text()); keys=('raw_path','hi_dev_path','hi_test_path'); missing=[c['data'][k] for k in keys if not Path(c['data'][k]).is_file()]; print('Missing files:', missing); raise SystemExit(bool(missing))"
git check-ignore data/raw outputs/check.pt .env config.local.yaml
git status --short
git log -1 --oneline
git rev-parse HEAD
```

预期：导入命令退出码 0，并打印 PyTorch 版本；设备命令打印实际设备；
路径检查输出 `Missing files: []` 并退出 0；忽略检查列出四个路径；
提交后 status 无输出，log 和 rev-parse 显示本地提交。
路径检查仅验证文件存在，不代表 T1 数据验证通过。

requirements.txt 列出当前实际使用的依赖，不锁定参考仓库的旧环境。
后续任务仅在实际需要时经确认增加依赖。

2026-10-10 实测：Python 3.11.15，PyTorch 2.5.1+cu121，NumPy 2.4.6，
pandas 3.0.3，h5py 3.16.0，SciPy 1.17.1，scikit-learn 1.9.1，PyYAML 6.0.3。
初次导入检查因缺少 scikit-learn 失败，安装该必需依赖后重验退出码为 0。
`torch.cuda.is_available()` 为 False，本次环境检查设备为 CPU；未验证 GPU 训练。
YAML 解析和三个路径检查退出码为 0，`Missing files: []`；四项忽略检查通过。
以上为 T0 环境验证记录；T1 真实数据验证结果见下文。未执行 smoke 或正式训练。
验收以交付回复中的完整 commit 为准，可用 `git log -1 --oneline` 查看。

## 数据位置与已批准配置

config.yaml 使用可移植相对路径。当前已发现：

| 用途 | 路径 | 当前检查范围 |
|---|---|---|
| 原始 DS03 | `data/raw/N-CMAPSS_DS03-012.h5` | X_s/W/A 字段、unit/cycle 和全量采样对齐通过 |
| dev HI 参考 | `data/raw/N-CMAPSS_DS03_cycle_dev.h5` | 663 条逐 flight 关联通过 |
| test HI 参考 | `data/raw/N-CMAPSS_DS03_cycle_test.h5` | 438 条逐 flight 关联通过 |

已有 `data/raw` 是本地符号链接，整个 data 目录被 Git 忽略；不提交链接目标的个人绝对路径。
当前项目路径和 Git 身份已提供，无缺少的 T0 环境信息。
若 T1 核验发现字段、设备编号或参考标签不符，先报告并向用户确认。

已批准配置见 config.yaml 与方案第 5 节：

| 项目 | 配置 |
|---|---|
| 数据 | 原始 dev/test；X_s 14 维、W 4 维；按 unit 连续每 10 点取 1 点；完整 flight |
| 模型 | 单层单向 GRU，X 隐藏 64、W 隐藏 32；d_I=d_C=8 |
| 解码与 HI | 时间条件 MLP 17→64→64→dx；HI 头 8→16→1；ReLU 隐藏层 |
| 通用 | seed=42；有 CUDA 则 CUDA，否则 CPU；最后一轮权重 |
| Stage 1 | Adam 默认其余参数，lr=0.001、weight_decay=0；batch=16；100 epochs；sigma_x²=1；KL 权重均为 1；梯度范数上限 5 |
| Stage 2 | Adam 默认其余参数，lr=0.001、weight_decay=0；batch=128 对；100 epochs；周期差≥5；kappa=1；显式含偏置 L2 系数 0.0001 |
| 排序抽样 | 均匀选有效设备，再均匀选该设备有效早晚对；允许重复；每 epoch 抽 N_train 对 |
| smoke | Stage 1 最多 32 个真实训练 flight；两阶段各 3 epochs；归一化由完整训练数据拟合；独立输出位置 |

SensorEncoder 只读 X/mask，W 只调制 z_C 先验，解码器只读潜变量与时间位置。
负 ELBO 先对每条 flight 的有效元素求和，再与 KL 相加，最后按 flight 平均。
Stage 2 冻结表示模块，`h=sigmoid(-g(mu_I))`。
HI/RUL 仅作评价；归一化与全局 HI Min–Max 仅由训练数据或训练预测拟合。
结构不保证表示解耦或物理尺度标定，不使用测试标签选模或早停。

当前已创建数据管线、模型、损失、Stage 1 训练和验证入口，已完成 Stage 1 smoke。
尚未实现 Stage 2 训练或评价指标，尚未运行正式训练。
本项目仅使用本地 Git，不配置远端，不 push。原始数据、缓存、权重、outputs 和凭据不提交。

## T1 运行与真实数据验证

在项目根目录执行：

```bash
python data.py --config config.yaml --prepare
python verify.py --task data --config config.yaml
```

预处理生成 `data/processed/ds03.pt`（Git 忽略），保存 dev/test flight 列表、列名、
数据设置与训练归一化参数。无 HI/RUL 标签，训练准备路径不读取两个参考表或原始 Y/T 字段。
`load_processed(config)` 加载预处理结果；`FlightDataset(processed, 'dev'/'test')`
返回 x/w/unit/cycle；`collate_fn` 输出 x/w/mask/unit/cycle，float32 序列按批次最长 flight
补零，bool mask 根据真实长度生成，unit/cycle 为 int64。

预期两条命令均退出 0，验证末行显示 `PASS: T1 real DS03 data verification (not smoke or training)`。
缺文件、配置不符、字段/关联不一致等错误返回非零退出码，不回退到合成数据。

| split | 设备 | flight 数 | 下采样后长度 | X/W 维数 |
|---|---|---|---|---|
| dev | 1–9 | 663 | 312–2029 | 14/4 |
| test | 10–15 | 438 | 312–2029 | 14/4 |

实际样本：dev `(1,1)` 长度 568，`(1,2)` 长度 312；test `(10,1)` 长度 1666，
`(10,2)` 长度 1773。命令还显示原始行数和首个采样行，便于核对跨 flight 连续采样相位。
没有引入验证集，dev/test 设备无重叠，unit/cycle 唯一、周期递增、无状态重置迹象。

用户对归一化问题回复“+是”，确认采用参考口径：X/W 各列在下采样前的全部原始 dev
时间点拟合 [0,1] Min–Max；test 原样使用训练参数，不裁剪。此确认体现在 config.yaml
的 normalization、normalization_fit 和 normalization_clip。实际没有常数列；常数列遵循
MinMaxScaler 的分母取 1 约定，不加 epsilon。读取按设备组织，每次只持有当前设备的原始 X/W，
避免同时载入全部无关 HDF5 数组。每 10 点取 1 点保持设备内相位，不在 flight 边界重启。

实际验证已核对每条 flight 的全部采样元素、X/W 共用行索引、长度、填充和 mask，
并独立从原始 dev 重算归一化极值。test 存在 3 个元素低于 0、46 个高于 1，均保留。
有效零值仍在 mask 内。全部训练批次只有约定的五个字段。

HI 表为 pandas fixed HDF 格式，直接由已有 h5py 读取数值块，不新增 PyTables 依赖。
`read_reference(path, flights)` 为独立评价读取函数，返回 unit/cycle/hi_gt。
表中没有显式 cycle；已检查存储行索引连续、设备行连续、每台设备行数与原始完整 flight
一致，且原始 cycle 恰为 1..N，随后按参考 Notebook 的设备内原始行顺序生成 cycle。
这属于经过核对的参考行顺序约定；文件没有额外周期字段可供独立交叉核验。
删除一条 flight 后关联被明确拒绝；配置与预处理设置不符时验证 CLI 实测返回退出码 1。

预处理实际执行两次，产物 SHA-256 均为
`b53a31a302e4aafafd635b38a4ad3fc1a9a863a9641d675a0402df7536cdaf17`，结果一致。
`python -m py_compile data.py verify.py` 及 Git diff 空白检查通过。
本次为真实数据验证，另含有效零值的 mask 检查；没有运行模型、smoke、训练或评价指标。

参考资料仅用于核对采样、归一化及 HI 顺序口径，代码独立实现。
记录的参考仓库 commit 为 `157ebfae7baeee1e0cd52180f8e58ead3dcb448e`：
[预处理 Notebook](https://github.com/KBaja/UnsupervisedHI/blob/157ebfae7baeee1e0cd52180f8e58ead3dcb448e/CMAPSS/01_data_preprocess.ipynb)、
[采样函数](https://github.com/KBaja/UnsupervisedHI/blob/157ebfae7baeee1e0cd52180f8e58ead3dcb448e/CMAPSS/pre_process_utils.py)。
仅沿用已确认的原始 dev/test 与完整 flight 路径。

## T1 曲线查看

用户要求在 verify.py 补充绘图任务，并回复“+同意”确认新增 matplotlib 依赖。
当前环境 matplotlib 3.10.9，绘图使用 Agg 后端输出 PNG，无需图形桌面。
绘制某一 flight 的指定 X_s/W 参数，以及该 unit 全部周期的参考 HI：

```bash
python verify.py --task data-plot --config config.yaml --split dev --unit 1 --cycle 1 --parameters T24 T30 Nf alt
```

预期退出 0，生成 `outputs/data_inspection/dev_unit_1_cycle_1.png`，
显示 568 个采样点的四条参数曲线和 unit 1 全部 72 个周期的 HI 曲线。
每个参数独立子图，纵轴为训练 Min–Max 归一化值，横轴为 flight 内的原始行位置，
保持每 10 点取 1 点的已确认采样口径；横轴不表示秒。
HI 是参考表的 ground truth，未经重新缩放或平滑，图中虚线标注所选 flight 周期。
此查看任务绘制参考 HI；训练后的模型预测曲线仍待后续任务实现。

参数名可选：`T24 T30 T48 T50 P15 P2 P21 P24 Ps30 P40 P50 Nf Nc Wf`
（X_s）或 `alt Mach TRA T2`（W）。必须显式指定 split/unit/cycle/parameters，
不存在的 flight、未知或重复参数、缺失必需参数均报错并返回非零退出码。
相同 split/unit/cycle 组合再次执行会覆盖该查看图，PNG 被 Git 忽略。

实际验证还运行：

```bash
python verify.py --task data-plot --config config.yaml --split test --unit 10 --cycle 66 --parameters T48 Wf Mach
```

退出 0，生成 `outputs/data_inspection/test_unit_10_cycle_66.png`，含 1251 个 flight 采样点和
unit 10 全部 66 个周期的参考 HI。绘图时检查曲线点数、坐标和值与输入数据一致；
已查看 dev 示例 PNG，标题、坐标和完整 HI 显示正常。`--parameters UNKNOWN` 实测退出码 1。
原有 `--task data` 已重验通过。未启动 T2 或模型训练。

## T2 模型与损失验证

```bash
python verify.py --task model --config config.yaml
```

预期退出 0，末行显示
`PASS: T2 model and loss verification (real forward/backward plus analytical checks; no training)`。
需要 T1 的 `data/processed/ds03.pt`；本任务不加载 HI/RUL 参考文件、不执行优化器更新或保存权重。
缺少数据或机制检查失败会返回非零退出码。

model.py 使用已批准的单层单向 GRU、四个后验线性头、两个条件先验线性头、时间条件 MLP
和 HI 头；全部采用 PyTorch 默认初始化。GRU 用真实长度打包序列，并取最后有效隐藏状态。
mask 必须是非空有效前缀和尾部 padding；时间位置按真实长度归一化，长度为 1 时为 0。
没有增加 dropout、log variance 裁剪、额外网络或损失。

```python
from model import OCALD, time_positions
model = OCALD(config)
mu_i, logvar_i, mu_c, logvar_c = model.encode(x, mask)
prior_mu_c, prior_logvar_c = model.conditional_prior(w, mask)
x_hat = model.decode(mu_i, mu_c, time_positions(mask, dtype=x.dtype))
hi = model.predict_hi(x, mask)  # sigmoid(-g(mu_i))，不读 W/cycle/标签
output = model(x, w, mask)     # 重参数化后验采样，供 Stage 1 目标使用
```

losses.py 提供逐 flight 有效元素求和的固定方差重构项、解析对角高斯 KL、按 flight 平均的
标准负 ELBO、早晚分数 softplus 排序项，以及包含头部权重和偏置的显式 L2。
重构在有效元素上运算，padding 不贡献损失；两个 KL 系数均为 1。

实际 CPU 验证：dev flight 索引 `[0, 1, 427]`，长度 `[568, 312, 2029]`；
X 形状 `(3, 2029, 14)`、W 形状 `(3, 2029, 4)`；未截断完整 flight。
为避免小验证批次的 CPU 线程开销，验证进程设单线程；未更改训练配置。

| 模块 | 参数量 |
|---|---|
| SensorEncoder | 17,440 |
| ConditionalPrior | 4,176 |
| Decoder | 6,222 |
| HIHead | 161 |
| 总计 | 27,999 |

随机初始化真实批次分项：重构 `1851.647461`，KL_I `0.084586`，KL_C `0.268159`，
总负 ELBO `1852.000366`。这是机制检查数值，不是训练、解耦或 HI 性能结果。
PyTorch float64 KL 参考最大绝对误差 `1.42e-14`，q=p 时为零。
手算小样本的重构为 `[2,1]`、KL_I 为 `[0.5,2]`，按 flight 平均的总目标为 `2.75`，检查通过。

验证还覆盖：替换 padding 为 NaN、增加 11 个 padding 时间步、单条无 padding 与批次对比，
有效后验/先验/解码/HI 与重构损失不变；时间位置首末为 0/1，单点 flight 为 0；
W 改变条件先验，但不改变后验、固定后验采样的重构或 HI；HI 用确定性 mu_I 与 sigmoid(-r)。
真实负 ELBO 反向能到达编码器、先验和解码器，HI 头不受该目标影响。
正确排序损失 `0.126928`，反向排序 `2.126928`，梯度方向正确；头参数的手算 L2
`144×2² + 17×1² = 593` 检查包含权重与偏置。

仅验证了 CPU，未验证 CUDA；未进行 smoke、正式训练、冻结训练或 checkpoint 保存/加载，
后两项留到对应训练任务。T2 验收后才启动 T3。

## T3 Stage 1 训练与 checkpoint

按已批准 smoke 预算运行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --config config.yaml --stage representation --smoke
```

CPU 环境设置上述线程数以减少小 GRU 批次的线程开销；不改变模型、batch、损失或训练预算。
实际设备按 CUDA 可用性选择，命令会报告设备、线程数、flight 数、模式与输出目录。
smoke 固定种子 42 抽取最多 32 个真实 dev flight，保持完整长度，归一化继续使用全部原始 dev
已拟合参数。每个 epoch 都覆盖选中 flight，训练批次打乱，日志按实际 flight 数加权平均。

每次运行创建新的时间戳目录 `outputs/representation_smoke_<UTC时间>/`；正式运行目录使用
`representation_formal_<UTC时间>`，不复用或覆盖已有目录。输出为 config.yaml、losses.csv 和
最后一个 epoch 的 representation.pt。CSV 独立记录 reconstruction、kl_i、kl_c 和总损失。

实际输出目录：`outputs/representation_smoke_20261010T084030540551Z/`。
验证这次产物：

```bash
python verify.py --task representation --config config.yaml --checkpoint outputs/representation_smoke_20261010T084030540551Z/representation.pt
```

预期退出 0，末行显示 `PASS: T3 representation training and checkpoint verification`。
重新训练时，将 checkpoint 路径替换为新命令实际打印的目录。

实际 CPU smoke：batch=16，32/663 条训练 flight，3 epochs，未加载 HI/RUL。

| epoch | 总损失 | 重构 | KL_I | KL_C |
|---|---|---|---|---|
| 1 | 1790.096497 | 1789.757690 | 0.080596 | 0.258116 |
| 2 | 1537.474365 | 1537.164368 | 0.084760 | 0.225291 |
| 3 | 1384.299072 | 1383.980225 | 0.114115 | 0.204688 |

三个表示模块更新的参数张量数分别为 12、8、6。HI 头未进入优化器、未更新，也未保存到表示权重中。
保存前后四个确定性后验输出（mu_I/logvar_I/mu_C/logvar_C）的最大绝对误差均为 0。
验证同时核对训练 subset、最后一轮 epoch、全训练归一化参数、列名和有限日志。
loss 不要求每个 KL 都下降，短程 loss 变化不能作为解耦或健康表征有效性的证据。

checkpoint 保存表示权重、必要配置、归一化、列名、阶段和 smoke/formal 信息；另保存
训练索引和三条 flight 的保存前后验，供本任务验证。没有保存 HI/RUL 标签。
`load_representation` 只需 checkpoint 即可初始化和加载表示模块：

```python
import torch
from train import load_representation
model, checkpoint = load_representation(checkpoint_path, torch.device('cpu'))
normalization = checkpoint['normalization']
```

加载后 model 为 eval 模式；该表示 checkpoint 中的 HI 头没有受训。
已在禁止 HDF5 读取的检查中独立加载并前向通过，不依赖原始数据文件或隐含脚本。
错误阶段 checkpoint 被拒绝，验证 CLI 实测退出码 1；编译和 Git diff 检查通过。

正式 Stage 1 入口已实现，沿用批准的 100 epochs、Adam 默认其余参数、lr=0.001、
weight_decay=0、batch=16、sigma_x²=1、两项 KL 权重 1 和梯度范数上限 5：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --config config.yaml --stage representation
```

本次只完成 smoke，**正式 100 epochs 未运行**，CUDA 未验证，Stage 2 未实现或启动。
正式评价前仍需完成正式 Stage 1→Stage 2；本次 smoke 结果仅作为训练与加载流程验证。
