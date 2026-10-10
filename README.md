# OCALD

首版范围为 PyTorch、N-CMAPSS DS03、完整 flight 输入和 flight 级 HI。
执行依据是 [AGENTS.md](AGENTS.md) 与 [实现方案 v2](OCALD_Implementation_Plan.md)。
每个任务完成验证及本地 Git 提交后，等待用户明确验收；设计批准不代表跨任务启动授权。

## 当前任务记录

| 任务 | 状态 | 验证摘要 | 用户确认记录 |
|---|---|---|---|
| T0 初始化 | 已验收 | 依赖、路径及忽略规则通过；commit b98f60d | 用户回复“开始T1”，确认继续 |
| T1 数据管线 | 已完成，待验收 | 真实数据全量索引、归一化、mask、HI 关联检查通过；重复预处理一致 | 用户已授权执行；已确认归一化口径；待验收 |
| T2 模型与损失 | 未开始 | 未验证前向、掩码或损失 | 设计与数值已批准；待 T1 验收 |
| T3 表示学习 | 未开始 | 未执行 smoke 或正式训练 | 设计与预算已批准；待 T2 验收 |
| T4 排序学习 | 未开始 | 未验证冻结、采样或推理 | 设计与预算已批准；待 T3 验收 |
| T5 评价 | 未开始 | 未计算评价指标 | 待 T4 验收 |

T0 只创建本 README、requirements.txt、.gitignore 和基础 config.yaml，
并将用户提供的两份规则与方案文件纳入首次本地提交。Python 实现文件按后续任务创建。
T1 新增 data.py 与 verify.py 的数据验证入口，更新数据配置和本文记录。

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

requirements.txt 列出 T0 必需依赖，不锁定参考仓库的旧环境。
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

当前已创建数据管线及数据验证入口，尚未创建模型、训练或评价指标实现，未运行任何训练。
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
