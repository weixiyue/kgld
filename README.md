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
| T3 表示学习 | 实现及 smoke 已验收 | Stage 1 smoke、checkpoint 通过；commit 2d8e4cb；正式训练未运行 | 用户回复“开始T4”，确认继续 |
| T4 排序学习 | 实现及 smoke 已验收 | 全部 663 条训练表示、3 epochs；采样、冻结、独立推理、训练缩放通过；commit c870ed6 | 用户回复“开始 T5”，确认继续 |
| T5 评价 | 实现与验证完成，待验收 | 正式 100/100 epochs、参考指标及 1101 条真实导出验证通过 | 用户授权 T5，并明确“先和参考实现对齐，参考实现代码中未考虑的问题暂时不用管”；待最终验收 |

T0 只创建本 README、requirements.txt、.gitignore 和基础 config.yaml，
并将用户提供的两份规则与方案文件纳入首次本地提交。Python 实现文件按后续任务创建。
T1 新增 data.py 与 verify.py 的数据验证入口，更新数据配置和本文记录。
T2 新增 model.py、losses.py，并扩展 verify.py 的模型机制检查。
T3 新增 train.py 的 Stage 1 入口与 checkpoint 加载，扩展 verify.py 的表示训练检查。
T4 扩展 train.py 的 HI 排序训练、完整 checkpoint 加载与缩放，并增加 verify.py 的 HI 检查。
T5 新增 metrics.py、evaluate.py，扩展 data.py 的独立评价 RUL 读取和 verify.py 的评价检查。

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

当前已创建数据管线、模型、损失、两阶段训练和评价入口，已完成两阶段 smoke。
T5 已完成正式训练、参考指标对齐和真实评价导出，等待最终验收，见下文。
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

## T4 HI 排序学习与独立推理

使用 T3 的表示 checkpoint 独立启动，不要求在同一进程先训练 Stage 1：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --config config.yaml --stage hi --representation outputs/representation_smoke_20261010T084030540551Z/representation.pt --smoke
```

实际交付输出目录：`outputs/hi_smoke_20261010T090923928503Z/`，包含 config.yaml、losses.csv、
最后一轮完整 ocald.pt。重新运行创建新的时间戳目录，不覆盖表示 checkpoint 或正式输出。
训练完整加载 Stage 1 表示模块并设 eval、requires_grad=False，提取全部 663 条 dev flight 的
mu_I 到内存，只优化 HI 头。训练路径不读取参考 HI 或 RUL，不用 cycle 预测 HI。

同设备周期排序及对构造在表示所在设备上执行。枚举所有周期差≥5的早晚对，然后均匀选
具有有效对的设备，再均匀选该设备的完整有效对，允许重复。设备对应有效对数为：
1:2278、2:2346、3:1953、4:1540、5:3916、6:1711、7:2850、8:2211、9:3160。
每个 epoch 抽取 N_train=663 对，batch=128 对，最后一个批次包含剩余 23 对。
没有有效对时明确报错；重复 unit/cycle 也会报错。

实际使用 CPU、Stage 1 smoke 权重、Stage 2 的 3 epochs smoke 预算：

| epoch | 总损失 | order | 未乘系数的 L2 |
|---|---|---|---|
| 1 | 0.693957 | 0.693275 | 6.823696 |
| 2 | 0.693864 | 0.693209 | 6.554293 |
| 3 | 0.693663 | 0.693032 | 6.309800 |

Adam lr=0.001、weight_decay=0；使用显式含偏置 L2，系数 0.0001，温度 1。
CSV 分项按实际样本对数加权汇总。表示参数及缓冲区训练前后逐值一致，保持 eval 且无梯度；
HI 头获得有限梯度并更新。训练前表示快照和头初始参数随 checkpoint 保存，供验证。

全部训练预测 `hi_raw` 范围为 `0.512035787..0.514756441`，变化很小。
全局 Min–Max 仅拟合这 663 条训练预测；未读取评价标签，未逐设备拟合，未平滑或裁剪。
该缩放是评价后处理，不能视为物理标定。极差为零时明确报告 scaled 不可用，
`scale_hi` 会报错，不添加 epsilon。当前范围虽非零，不能据此认定健康表征有效。

独立验证：

```bash
python verify.py --task hi --config config.yaml --checkpoint outputs/hi_smoke_20261010T090923928503Z/ocald.pt
```

预期退出 0，末行显示 `PASS: T4 HI training and independent inference verification`。
检查包含：真实对的设备/间隔/索引、冻结快照与最终状态完全一致、头参数更新与梯度、
仅 X/mask 推理、原始 HI 在 (0,1)、重复预测一致、保存/加载一致、完整训练归一化和
仅训练预测拟合的 Min–Max；缩放超范围值仍保留，无有效对及零极差被明确拒绝。
另用一个打乱周期且早期 flight 候选数不同的小例子，独立枚举对并做固定种子抽样检查，
检验均匀设备及均匀有效对分布，防止误用“均匀早期 cycle 再选晚期 cycle”的分布。

独立加载无需原始 HDF5、参考表、表示源 checkpoint 或预处理文件：

```python
import torch
from train import load_ocald, scale_hi
model, checkpoint = load_ocald(checkpoint_path, torch.device('cpu'))
# x 使用 checkpoint['normalization']['x'] 进行训练参数归一化。
with torch.no_grad():
    hi_raw = model.predict_hi(x, mask)
    hi_scaled = scale_hi(hi_raw, checkpoint['hi_minmax'])
```

已在禁止 HDF5 访问的独立加载检查中，用单点合成输入仅传 X/mask 推理通过；
该合成输入检查用于证明接口独立，真实验证另使用全部训练 flight。
完整 checkpoint 保存模型、训练归一化、列名、阶段、来源模式/epoch 和训练 HI Min–Max。
正式 Stage 2 要求正式表示 checkpoint；将 smoke 表示传给正式入口实测退出码 1，未启动训练。
原有 T3 checkpoint 验证、编译及 Git diff 检查也通过。

正式 Stage 2 入口已实现，预算 100 epochs；本次只执行 smoke，**两阶段正式训练均未运行**，
CUDA 未验证，未计算评价指标或启动 T5。正式评价前仍须先完成正式 Stage 1→Stage 2。

## T5 评价与端到端运行

以上 T0–T4 的 CPU 与未执行正式训练记录是对应任务交付时的历史状态。
T5 发现沙箱内 CUDA 不可见，宿主环境实际有两张 RTX 3090；正式运行按已批准 auto
配置使用 cuda（单卡，无分布式）。未完成的 CPU 尝试保留在
`outputs/representation_formal_20261010T092156715251Z/`，不作为正式模型。
完整预算从头在 GPU 执行，没有改变网络、损失、batch、flight 长度或初始化规则。
CUDA 模型机制验证已通过，包括真实前向/反向、padding、信息路径和解析损失检查。

完整运行命令如下。两阶段每次创建新的时间戳目录，将占位路径替换为上一条命令的实际输出。

```bash
python data.py --config config.yaml --prepare
python verify.py --task data --config config.yaml
python verify.py --task model --config config.yaml
python train.py --config config.yaml --stage representation
python verify.py --task representation --config config.yaml --checkpoint outputs/<representation_run>/representation.pt
python train.py --config config.yaml --stage hi --representation outputs/<representation_run>/representation.pt
python verify.py --task hi --config config.yaml --checkpoint outputs/<hi_run>/ocald.pt
python verify.py --task metrics --config config.yaml
python evaluate.py --config config.yaml --checkpoint outputs/<hi_run>/ocald.pt
python verify.py --task evaluation --config config.yaml --run-dir outputs/<hi_run>
```

正式预算为 Stage 1/Stage 2 各 100 epochs，只保存最后一轮，不用评价标签选模。
本次 GPU Stage 1 目录为 `outputs/representation_formal_20261010T102117772658Z/`。
仅完成 smoke 时，必须在报告与曲线中保留 smoke 标记，不能作为正式实验结果。

评价输出写入对应完整 checkpoint 所在目录：

| 文件 | 内容 |
|---|---|
| predictions.csv | dev/test 全部 1101 条 flight；split/unit/cycle/hi_raw/hi_scaled/hi_gt/rul |
| metrics.json | raw/scaled 整体及适用的逐设备指标、有效计数、模式和来源轮数、训练缩放参数 |
| hi_curves.png | 全部 15 台设备的完整 raw/scaled 曲线、参考 HI 和逐设备指标 |

HI/RUL 按 unit/cycle 一对一关联，缺失、额外或重复键会报错，禁止位置拼接。
原始 RUL 已核对 1101 条 flight 内数值恒定且非负；独立 `read_rul` 每次重新检查这一事实，
不满足时报错。训练准备、Stage 1 和 Stage 2 调用路径不读取这些评价量。
参考 HI 表无 cycle 字段的限制与 T1 相同，仍使用已核对的设备内行顺序约定。

| 指标 | 参考口径 |
|---|---|
| MAE | 对所有 flight 的绝对误差求均值 |
| MAPE (%) | 排除 HI_gt=0，再按 flight 汇总；报告有效数量 |
| 单调性 | 设备内正/非正差分计数差的绝对值除以差分数；零差分归非正；整体为设备均值 |
| 趋势性 | 设备内 HI/cycle Spearman；整体先求有符号均值再取绝对值 |
| 失效一致性 | exp(-std(各设备末 HI, ddof=0)/mean(各设备首 HI−末 HI))；仅整体 |
| MI | 设备内 sklearn 默认 3 邻居估计 HI/RUL 互信息，seed=42；取 1−exp(-MI)，整体为设备均值 |

非退化手算例与固定参考 commit 的六项公式核对通过；表行打乱不改变结果。
2026-10-12 用户明确要求先对齐参考实现，参考代码未考虑的问题暂时不用管。
因此撤去额外的小样本、常数 HI/RUL 和单设备/零分母拦截，未增加 null 汇总约定。
常数 HI 单调性为 1；Spearman 的 NaN 按参考实现转为 0；MI 直接调用 sklearn，
仍保留已批准的 seed=42；单设备失效一致性直接按参考公式计算。
验证覆盖常数 HI、常数 RUL 的 MI、Spearman NaN 转 0、单设备公式和原有手算/关联检查。
参考未定义保护的空 MAPE、短序列、零分母等不增加特殊策略：底层函数可能报错或返回
非有限数，JSON 导出仍要求有限数字。当前正式数据未触发这些情况，不声称它们已获额外处理。
训练 HI 零极差按已批准方案使 scaled 不可用，不加 epsilon；测试不重新拟合、裁剪或平滑。
失效一致性若因平均首末差为负而超过 1，保留参考公式结果并解释，不悄悄改公式。

CSV 使用 17 位有效数字并以 round_trip 读取，指标及曲线都由导出的同一表生成，
避免 float32 指标计算与 CSV 的 float64 重算产生差异。验证重新推理并检查实际行数、
参考关联、CSV 数值、双尺度、指标一致性以及曲线点值；失败返回非零退出码。
训练 Min–Max 只是评价后处理，曲线和指标不能证明潜变量解耦或物理尺度标定。

已完成 smoke 导出验证：使用 `outputs/hi_smoke_20261010T090923928503Z/ocald.pt`，
输出 663 条 dev 和 438 条 test；预测、标签、CSV、指标与曲线一致。
正式运行输出：

| 阶段 | 输出目录 | 实际结果 |
|---|---|---|
| Stage 1 | `outputs/representation_formal_20261010T102117772658Z/` | CUDA，全部 663 flight，100 epochs；最后负 ELBO 29.073746，重构 28.050419，KL_I 0.062563，KL_C 0.960764 |
| Stage 2 | `outputs/hi_formal_20261010T104138118262Z/` | CUDA，100 epochs；最后总损失 0.687845，order 0.687127，未乘系数 L2 7.183465 |

正式表示 checkpoint 验证通过，保存/加载四组后验最大绝对误差为 0。
正式 HI checkpoint 验证通过：冻结权重与缓冲区逐值一致、头更新、均匀采样、独立推理和训练缩放均通过。
正式预测导出及评价验证通过，663 dev/438 test，与重新推理、参考标签、CSV 指标和曲线点值一致。
最后一轮训练 raw HI 极值为 0.410069942..0.568396688；test scaled 范围为
−0.035984..0.999826，未裁剪。正式图已查看，15 台设备编号和指标均显示。

| split / 尺度 | MAE | MAPE (%) / 有效数量 | 单调性 | 趋势性 | 失效一致性 | MI |
|---|---|---|---|---|---|---|
| dev raw | 0.340664 | 71.313839 / 654 | 0.045597 | 0.114314 | 4.171531e-08 | 0.036016 |
| dev scaled | 0.384121 | 75.278862 / 654 | 0.045597 | 0.114314 | 4.171470e-08 | 0.036016 |
| test raw | 0.323512 | 70.778683 / 432 | 0.051596 | 0.057376 | 0.094559 | 0.015828 |
| test scaled | 0.355946 | 68.945694 / 432 | 0.051596 | 0.057376 | 0.094559 | 0.015828 |

上述为完整批准预算的单次正式运行，不是 smoke；单调性和趋势性较弱，不能认为 HI 学习效果良好。
没有根据测试指标修改模型、训练或后处理；未开展多种子、其他数据集、基线或消融实验。

直接重验本次正式产物：

```bash
python verify.py --task representation --config config.yaml --checkpoint outputs/representation_formal_20261010T102117772658Z/representation.pt
python verify.py --task hi --config config.yaml --checkpoint outputs/hi_formal_20261010T104138118262Z/ocald.pt
python verify.py --task metrics --config config.yaml
python evaluate.py --config config.yaml --checkpoint outputs/hi_formal_20261010T104138118262Z/ocald.pt
python verify.py --task evaluation --config config.yaml --run-dir outputs/hi_formal_20261010T104138118262Z
```

预期均退出 0；指标末行显示 `PASS: T5 reference-aligned metrics, keyed associations and train-only scale availability`，
评价末行显示 `PASS: T5 evaluation verification (predictions, references, metrics and curves)`。
2026-10-12 参考对齐修改后指标检查、正式重新导出及评价检查均退出 0；
predictions.csv 和 metrics.json 的 SHA-256 与修改前一致，正式指标没有改变；无需重新训练。
代码编译和 Git diff 空白检查通过；数据、权重、输出均未提交。
T5 实现与验证完成，等待用户对最终本地 commit 明确验收，不自动扩展实验。
