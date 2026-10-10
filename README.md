# OCALD

首版范围为 PyTorch、N-CMAPSS DS03、完整 flight 输入和 flight 级 HI。
执行依据是 [AGENTS.md](AGENTS.md) 与 [实现方案 v2](OCALD_Implementation_Plan.md)。
每个任务完成验证及本地 Git 提交后，等待用户明确验收；设计批准不代表跨任务启动授权。

## 当前任务记录

| 任务 | 状态 | 验证摘要 | 用户确认记录 |
|---|---|---|---|
| T0 初始化 | 已完成，待验收 | 七项依赖导入通过；YAML 和三个文件路径检查通过；忽略规则通过；交付对应本地 T0 提交 | 用户已授权执行；尚未确认验收 |
| T1 数据管线 | 未开始 | 未验证真实数据字段、flight 或 HI 关联 | 设计与数值已批准；待 T0 验收及启动批准 |
| T2 模型与损失 | 未开始 | 未验证前向、掩码或损失 | 设计与数值已批准；待 T1 验收 |
| T3 表示学习 | 未开始 | 未执行 smoke 或正式训练 | 设计与预算已批准；待 T2 验收 |
| T4 排序学习 | 未开始 | 未验证冻结、采样或推理 | 设计与预算已批准；待 T3 验收 |
| T5 评价 | 未开始 | 未计算评价指标 | 待 T4 验收 |

T0 只创建本 README、requirements.txt、.gitignore 和基础 config.yaml，
并将用户提供的两份规则与方案文件纳入首次本地提交。Python 实现文件按后续任务创建。

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
当前仅完成 T0 环境验证，未进行合成数据验证、真实数据内容验证、smoke 或正式训练。
验收以交付回复中的完整 commit 为准，可用 `git log -1 --oneline` 查看。

## 数据位置与已批准配置

config.yaml 使用可移植相对路径。当前已发现：

| 用途 | 路径 | T0 检查范围 |
|---|---|---|
| 原始 DS03 | `data/raw/N-CMAPSS_DS03-012.h5` | 文件存在，内容待 T1 核对 |
| dev HI 参考 | `data/raw/N-CMAPSS_DS03_cycle_dev.h5` | 文件存在，关联待 T1 核对 |
| test HI 参考 | `data/raw/N-CMAPSS_DS03_cycle_test.h5` | 文件存在，关联待 T1 核对 |

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

当前尚未创建数据、模型、训练或评价实现，未运行任何训练。
本项目仅使用本地 Git，不配置远端，不 push。原始数据、缓存、权重、outputs 和凭据不提交。
