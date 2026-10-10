# OCALD：DS03 精简实现方案

版本：v2，2026-10-10。T1–T4 的设计、网络结构和数值配置已获用户确认；采用本地 Git，保留逐任务验收。

## 1. 使用方式与任务边界

本文件交给 Codex 逐任务执行，配套 `AGENTS.md` 放在项目根目录。当前交付仅为方案和执行规则，不代表已生成、运行或验证模型代码。

执行顺序：T0 → T1 → T2 → T3 → T4 → T5。每次只做当前获准任务；完成验证、本地 Git 提交后，向用户交付可重复运行的验证接口，等待用户明确确认，才可开始下一任务。“全部实现完再统一验收”不符合本方案。

用户提出当前任务的修复要求时，只修复当前任务，重新验证并提交本地 Git，再等待确认。任务通过后若需要改变已确认接口或方法，应先说明原因并获得确认。

首版范围：PyTorch、N-CMAPSS DS03、完整 flight 输入、flight 级 HI、两阶段训练、参考仓库评价指标。暂不实现其他数据集、基线模型、消融调度、联合微调、条件流、分布式训练、下游 RUL 预测或自动超参数搜索。

## 2. 已确认的方法约定

| 项目 | 实现约定 |
|---|---|
| 样本 | 一个设备的一个 flight；以 `(unit, cycle)` 唯一标识 |
| 输入 | 传感器序列 X、工况序列 W、有效时间步 mask |
| 传感器编码器 | 只接收 X 和 mask，共享时序特征后输出两组对角高斯后验 |
| 内部块先验 | p(z_I) = N(0, I) |
| 工况路径 | 完整 W 经小型时序编码器生成 u，再输出 z_C 条件先验参数 |
| 条件先验 | p(z_C | W, mask) 为对角高斯；首版使用仿射条件变换 |
| 解码器 | 接收 z_I、z_C 和归一化时间位置，输出传感器重构；不接收 W 或 u |
| Stage 1 | 标准负 ELBO，两个 KL 系数均为 1 |
| Stage 2 | 冻结表示模型，以 mu_I 训练标量头；h = sigmoid(-r) |
| 时间信息 | 仅用于同设备内的排序监督，不作为编码器或 HI 头输入 |
| HI/RUL | 只用于评价；不参与训练、选模或早停 |
| 输出尺度 | 同时保留原始 Sigmoid HI 和训练集 Min–Max 后处理 HI |

条件先验提供结构约束，不自动证明潜变量实现了工况解耦。排序和 Sigmoid 不提供物理端点标定。代码、README 和验证报告不得夸大这些性质。

### 2.1 损失的精确定义

对单个 flight，固定方差高斯重构项为：

```math
L_{rec} = \frac{1}{2\sigma_x^2}\sum_{t:m_t=1}\sum_j (x_{tj}-\hat x_{tj})^2.
```

先对有效时间步和传感器维度求和，再将每个 flight 的重构项与两个 KL 相加，最后在批次内对 flight 求平均。不得仅将重构项改成逐元素平均，同时保留原来的 KL 权重；这会改变现有方法稿的目标。可以另行记录逐元素重构误差，供观察使用。

对角高斯 KL 使用解析式。对 q=N(mu_q,diag(v_q)) 和 p=N(mu_p,diag(v_p))：

```math
D_{KL}(q\|p)=\frac12\sum_k\left[\log\frac{v_{p,k}}{v_{q,k}}+\frac{v_{q,k}+(\mu_{q,k}-\mu_{p,k})^2}{v_{p,k}}-1\right].
```

后验训练采样采用重参数化。HI 推理使用 mu_I，不使用随机 z_I。

同设备早晚 flight 对 a<b 的排序目标为：

```math
L_{ord}=\operatorname{mean}_{(i,a,b)}\operatorname{softplus}\left(\frac{r_{i,a}-r_{i,b}}{\kappa}\right),\qquad
L_{HI}=L_{ord}+\lambda_\xi\sum_{p\in\xi}\|p\|_2^2.
```

显式计算 HI 头参数（包含偏置）的 L2 项；不能用 AdamW 的解耦 weight decay 冒充这个目标，也不能重复施加两份正则。温度、正则强度和样本对选择按第 5 节已批准配置执行。

## 3. 项目结构

核心实现平铺在根目录，不创建通用框架。

| 路径 | 内容 |
|---|---|
| `AGENTS.md` | Codex 执行规则、确认门槛和 本地 Git 提交要求 |
| `OCALD_Implementation_Plan.md` | 本方案 |
| `README.md` | 环境、运行命令、已确认配置、任务状态与验收记录 |
| `requirements.txt` | 实际使用的依赖；不沿用参考仓库的旧环境锁定 |
| `.gitignore` | 排除原始数据、缓存、权重、运行输出和凭据 |
| `config.yaml` | 一个 DS03 配置；不建立多层配置继承 |
| `data.py` | 预处理、FlightDataset、collate_fn 和数据检查入口 |
| `model.py` | SensorEncoder、ConditionalPrior、Decoder、HIHead、OCALD |
| `losses.py` | 重构项、KL、排序和 L2 |
| `train.py` | Stage 1、提取 mu_I、Stage 2 与训练命令入口 |
| `metrics.py` | 独立的 NumPy/SciPy/scikit-learn 评价函数 |
| `evaluate.py` | 预测、参考值关联、指标和绘图 |
| `verify.py` | 用户要求的统一验证接口，仅包含任务所需的关键正确性检查 |
| `data/raw/` | 原始 DS03 和两个参考 HI 文件，不进入 Git |
| `data/processed/` | 预处理结果，不进入 Git |
| `outputs/<run_name>/` | 一次运行的结果，不进入 Git |

`verify.py` 是为逐任务验收增加的唯一辅助脚本；不再另建测试平台、任务调度器或复杂报告系统。不需要空目录占位、空实现文件、模型注册器、通用 Trainer、历史身份系统或缓存版本协议。按任务逐步创建真正用到的文件。

## 4. 数据与模块接口

### 4.1 DS03 数据流

参考仓库使用 `N-CMAPSS_DS03.h5` 以及 `N-CMAPSS_DS03_cycle_dev.h5`、`N-CMAPSS_DS03_cycle_test.h5`。实际本地文件名和路径以用户提供为准，不能只凭 DS003 的称呼推断。

读取原始文件中的传感器、工况和辅助字段，依据变量名核对 unit/cycle 列；按原始 dev/test 划分组织数据。归一化参数仅由训练部分拟合。任何验证集若被引入，应先按设备划分，再拟合归一化参数，不得在同一设备周期之间随机切分训练和验证。

参考 Notebook 有 ID/OOD 和完整 flight/短窗口等可选单元格，不能按线性执行结果照搬。首版只选择已确认的 DS03 dev/test 和完整 flight 路径。

传感器与工况必须使用相同的行选择与下采样索引。已确认沿用参考代码每 10 点取 1 点、按 unit 连续采样，使用 X_s 传感器变量和 W 工况变量。预期分别为 14 和 4 维；实际字段若不符，先核对并询问，不自行筛列。不得静默改为逐 flight 重新取样，也不得无说明截断长 flight。

训练样本不含 HI ground truth 或 RUL。数据代码可独立提供评价参考值读取函数，但训练调用路径不得加载它们。

```python
batch = {
    'x': ...,       # float32 [B, T_max, dx]
    'w': ...,       # float32 [B, T_max, dw]
    'mask': ...,    # bool [B, T_max]
    'unit': ...,    # int64 [B]
    'cycle': ...,   # int64 [B]
}
```

mask 来自真实长度，不能通过 x==0 推断。禁止额外要求 N-CMAPSS 不具备的 `history_id`；同设备的已提供周期序列就是当前排序组织依据。若实际数据出现重复编号或明确的状态重置，报告事实并询问，不臆造字段。

预处理结果可使用一个文件保存训练和测试 flight 列表、列名与训练归一化参数。首次实现只保留这一种形式。原始文件读取方式需根据实际体量选择；避免同时复制所有无关 HDF5 数组，也不预先建设流式存储平台。

### 4.2 模型接口

```python
model.encode(x, mask)                  # mu_i, logvar_i, mu_c, logvar_c
model.conditional_prior(w, mask)       # prior_mu_c, prior_logvar_c
model.decode(z_i, z_c, time_positions) # x_hat
model.predict_hi(x, mask)             # [B]，确定性的原始 HI
```

时间位置按每条 flight 的有效长度归一化；长度为 1 时令其位置为 0。填充位置不参与池化、时序状态提取或重构损失。不同 padding 长度不应改变同一 flight 的确定性 HI。

这些是目标接口，不强制建立额外结果类。普通字典和张量足够。网络骨干和层数已在第 5 节确认，不能自行加入注意力、Transformer、对抗损失或额外分支。

### 4.3 训练与保存

Stage 1 优化传感器编码器、条件先验和解码器。Stage 2 对表示模块设置 eval、关闭梯度并仅优化 HI 头；可以一次提取 mu_I 到内存，当前版本不需要独立 latent cache 管理协议。

Stage 2 必须能通过 `representation.pt` 单独启动，不能要求先运行某个隐含脚本。checkpoint 保存必要模型配置、归一化参数、已训练权重和当前阶段；不开发跨任意配置的自动恢复机制。

推理只需 X、mask 和训练归一化参数，不需要 W、cycle、HI ground truth 或 RUL。训练日志分别记录 reconstruction、KL_I、KL_C、order、L2，不只记录总损失。

### 4.4 评价与双尺度输出

按 `(unit, cycle)` 关联预测与参考值。参考 Notebook 从 HI 表的每台设备行顺序生成周期编号；若实际文件没有 cycle 字段，必须核对行数及排序与原始 flight 列表一致。无法可靠对齐时停止误差评价，不做位置拼接。

原始输出记为 `hi_raw`。用训练 flight 的 `hi_raw` 拟合一个全局 Min–Max 变换，应用到测试预测得到 `hi_scaled`；不使用 HI 标签，不逐设备拟合，也不使用测试极值。测试 scaled 值可能超出 [0,1]，不默认裁剪。训练输出极差为零时将 scaled 指标标为不可用并解释原因，不加任意 epsilon 伪造有效尺度。

分别报告 raw/scaled 两套指标与曲线；Min–Max 只是参考评价后处理，不描述为物理标定。保持参考实现的 MAE/MAPE 汇总方式；MAPE 排除 HI_gt=0 的样本，并报告有效样本数。

参考指标的关键口径：单调性将零差分归入非正差分；趋势性先对各设备 Spearman 相关系数取平均再取绝对值；失效一致性使用各设备首末 HI；MI 使用 RUL 作为评价量并按参考变换汇总。MI 固定随机种子。小样本、常数序列和零分母需明确处理，不能隐藏 NaN 或默默改公式。常数 HI 即使在某个指标上得高分，也不得解释为健康表征有效。

输出：`config.yaml`、`representation.pt`、`ocald.pt`、`losses.csv`、`predictions.csv`、`metrics.json`、`hi_curves.png`。预测表至少包含 split、unit、cycle、hi_raw、hi_scaled、hi_gt；若计算 MI，关联 rul。metrics 保存整体和适用的逐设备指标；失效一致性等跨设备指标不硬套逐设备解释。

## 5. 已确认配置与执行时信息

以下内容已获用户批准，直接实现，不重复询问。改变这些选择仍需用户确认。

### 5.1 T2 网络

| 模块 | 已确认结构 |
|---|---|
| SensorEncoder | 单层单向 GRU，隐藏维度 64；取最后一个有效隐藏状态；四个线性头输出 mu_I、logvar_I、mu_C、logvar_C |
| 潜变量 | d_I=8，d_C=8 |
| ConditionalPrior | 单层单向 GRU，隐藏维度 32，编码完整 W；取最后一个有效隐藏状态；两个线性头输出条件先验均值和 log variance |
| Decoder | 每个时间位置共享 MLP：[z_I,z_C,s_t]，17→64→64→dx；两个隐藏层 ReLU，输出层线性 |
| HIHead | 8→16→1，隐藏层 ReLU，输出 r；h=sigmoid(-r) |

GRU 按真实长度处理变长序列，不取 padding 末尾状态。使用标准 PyTorch 层默认初始化，不额外加入 dropout、归一化层或复杂初始化。会改变分布或损失的数值处理（如 log variance 裁剪）需先确认，不在异常后静默添加。

### 5.2 T3/T4 训练

| 阶段 | 参数 | 已确认值 |
|---|---|---|
| 通用 | 随机种子 | 42 |
| 通用 | 设备 | 有 CUDA 则使用，否则 CPU；报告实际设备 |
| T3 | 优化器 | Adam，weight_decay=0，其余参数使用默认值 |
| T3 | 学习率 | 0.001 |
| T3 | 批大小 | 16 个 flight |
| T3 | 正式预算 | 100 epochs |
| T3 | sigma_x² | 1.0，固定 |
| T3 | KL_I / KL_C 权重 | 1 / 1 |
| T3 | 梯度裁剪 | 全局梯度范数上限 5 |
| T4 | 优化器 | Adam，weight_decay=0，其余参数使用默认值 |
| T4 | 学习率 | 0.001 |
| T4 | 批大小 | 128 对 flight |
| T4 | 正式预算 | 100 epochs |
| T4 | 最小周期差 | cycle_b - cycle_a ≥ 5 |
| T4 | 温度 kappa | 1.0 |
| T4 | 显式 L2 系数 | 0.0001，包含 HI 头偏置 |

T4 先均匀抽取具有有效对的训练设备，再均匀抽取该设备满足间隔要求的早晚对，允许重复抽样。每个 epoch 抽取 N_train 对，N_train 为训练 flight 总数。注意：均匀选早期周期再均匀选其可用晚期周期，并不等于均匀抽取有效对；不要偷换抽样分布。实现保持简单，不建立采样框架。

两阶段均使用最后一个 epoch 的权重，不引入早停、调度器、KL 退火或按测试指标选模。固定随机种子，不声称跨硬件逐位一致。

### 5.3 快速验证和正式运行

- Stage 1 smoke：种子 42 固定抽取最多 32 个真实训练 flight，运行 3 epochs；归一化仍由完整训练部分拟合。
- Stage 2 smoke：使用全部训练 flight 的冻结表示，按正式采样规则运行 3 epochs。可以使用 Stage 1 smoke 权重检查流程，但必须标注 smoke。
- smoke 使用独立运行名，报告其实际路径，不能覆盖正式输出。
- 正式预算已批准为每阶段 100 epochs；只在对应任务获得启动批准后执行。T3/T4 如仅完成 smoke，应明确标注，T5 正式评价前仍须完成正式 Stage 1→Stage 2，不把短程指标当论文结果。
- 显存不足、非有限损失或无法满足计算预算时说明情况并询问，不静默减小 batch、截断 flight 或更改训练目标。

### 5.4 剩余环境信息

方法和初始超参数已无待确认项。用户在本地提供项目目录，以及原始 DS03 和两个参考 HI 文件的路径即可。已有 Git 仓库使用当前分支；新项目在用户指定目录初始化本地 Git。提交身份缺失时询问，不伪造姓名邮箱。不需要远端地址，不 push。

实际数据与预期不同、出现异常或需要改变已批准设计时，再提出具体问题；这些情形不授权 Codex 自行补充新方法。

## 6. 分任务实现和验收

以下命令是**要求 Codex 实现的目标验证接口**，当前文档交付时尚不存在，不表示已运行通过。实现后必须在实际环境执行并报告结果。验证命令失败应返回非零退出码，不得只打印警告后 PASS。

### T0：项目初始化与规则落地

前提：用户在本地指定项目位置；检查已有仓库，新项目初始化本地 Git。无需远端。缺少提交身份时询问用户。

实现：建立精简目录，放入两份方案文件；编写 README、requirements、.gitignore 和基础 config。记录 T0–T5 状态、已批准配置和缺少的环境信息；不提前创建模型实现。检查 PyTorch 和必要数据依赖可导入，按用户机器确认执行设备。

验证接口：

```bash
python -c "import torch, numpy, pandas, h5py, scipy, sklearn, yaml; print(torch.__version__)"
git status --short
git log -1 --oneline
git rev-parse HEAD
```

通过条件：依赖可导入；有正确 .gitignore；本次修改已进入本地 commit，工作区无未解释的残留修改；没有数据、权重或凭据被提交。报告环境及 commit，等待用户批准 T1。

### T1：真实 DS03 数据管线

前提：T0 已验收，实际数据路径已提供；按第 5 节执行已批准的数据选择。

实现：只完成 data.py、数据相关配置和 verify.py 的数据检查；按完整 flight 建立 Dataset 和 collate；独立读取参考 HI 并检查关联，但训练 batch 不含评价标签。

验证接口：

```bash
python data.py --config config.yaml --prepare
python verify.py --task data --config config.yaml
```

关键验证：真实 dev/test 设备集合及重叠检查；每个 split 的 flight 数、长度范围和输入维度；unit/cycle 唯一性；X/W 对齐；mask 有效长度；训练拟合的归一化参数原样用于测试；HI 与 flight 一一对应；变长 flight 不被静默截断。向用户展示少量真实样本元信息，不输出整批原始数组。

通过条件：实际数据检查通过，预处理能重复运行；报告数据选择和关联结果。仅有合成数据检查不能算真实 T1 通过。本地 Git 提交后等待用户批准 T2。

### T2：模型前向与损失

前提：T1 已验收，网络结构与规模按第 5 节已批准配置执行。

实现：model.py 和 losses.py；扩展 verify.py。不开始正式训练。

验证接口：

```bash
python verify.py --task model --config config.yaml
```

关键验证：一个真实小批次的输入输出维度正确，损失有限；KL 与 PyTorch 分布解析结果一致且 q=p 时接近零；替换填充数值或增加 padding 不改变有效输出；反向传播能到达编码器、条件先验和解码器；改变 W 不影响 encode(X) 和 predict_hi(X) 的结果；早期 r 小于晚期 r 时排序损失比反向更低。展示参数量和分项损失。

通过条件：上述机制检查通过。随机初始化输出不用于评价模型性能。本地 Git 提交后等待用户批准 T3。

### T3：Stage 1 表示学习

前提：T2 已验收，似然方差、训练参数和选模规则按第 5 节已批准配置执行。

实现：train.py 的表示学习入口、日志和 representation.pt 保存/加载；不训练 HI 头。

验证接口：

```bash
python train.py --config config.yaml --stage representation --smoke
python verify.py --task representation --config config.yaml --checkpoint outputs/<run_name>/representation.pt
```

`--smoke` 仅运行 config 中已确认的小训练预算，使用独立运行名，日志明确标注 smoke，不能覆盖正式输出。检查参数确实更新、损失有限、checkpoint 可独立加载、保存前后确定性后验均值一致；显示实际重构与两个 KL 的量级。

通过条件：训练与加载链路正确。必要时在用户同意的预算内做小样本拟合检查；不要求每个 KL 同时下降，不把短程 loss 下降视为解耦证据。本地 Git 提交后等待用户批准 T4。

### T4：Stage 2 排序学习与独立推理

前提：T3 已验收，采样策略和 HI 头训练参数按第 5 节已批准配置执行。

实现：从 representation.pt 加载并冻结表示模型，提取 mu_I，构造同设备早晚对，只训练 HI 头；保存完整 ocald.pt 和训练 HI 的 Min–Max 参数。不引入 history_id 或中间文件管理平台。

验证接口：

```bash
python train.py --config config.yaml --stage hi --representation outputs/<run_name>/representation.pt --smoke
python verify.py --task hi --config config.yaml --checkpoint outputs/<run_name>/ocald.pt
```

关键验证：样本对同设备且顺序、间隔正确；无有效对时清楚报错；表示模型参数和缓冲区训练前后完全一致；HI 头获得梯度并更新；推理只用 X 和 mask；原始 HI 在 (0,1) 内、同输入结果可重复；Min–Max 参数仅来自训练预测，不读取参考 HI。

通过条件：损失方向、冻结行为和独立加载推理正确。短程输出即使接近常数也如实报告，不能加入未经确认的端点损失“修复”它。本地 Git 提交后等待用户批准 T5。

### T5：评价、曲线和端到端验收

前提：T4 已验收；正式训练使用已批准预算；若需扩大，先确认。

实现：metrics.py 和 evaluate.py；输出 raw/scaled 两套结果、逐设备曲线及总体汇总；完善 README 中的完整运行命令。

验证接口：

```bash
python verify.py --task metrics --config config.yaml
python evaluate.py --config config.yaml --checkpoint outputs/<run_name>/ocald.pt
python verify.py --task evaluation --config config.yaml --run-dir outputs/<run_name>
```

关键验证：用少量手算可核对的序列验证 MAE、MAPE、单调性和趋势性；核对参考实现的非退化输入结果；覆盖常数序列、零参考值、缺失关联和零尺度；MI 固定随机种子；CSV 行数和真实 flight 数一致；曲线数据与 CSV 一致；双尺度输出没有测试集重新拟合或裁剪。每条测试设备曲线标注设备编号和指标。

通过条件：上述评价检查通过，用户能用 README 从数据准备运行到结果导出。smoke 指标必须明确标为流程验证结果；没有完成正式训练时不能把它写成论文最终实验。本地 Git 提交后提交最终验收，不自动扩展到其他实验。

## 7. 每次交付给用户的固定内容

1. 本次任务编号、完成的功能和修改文件。
2. 实际运行的验证命令、结果、输出位置及未验证项。
3. 用户可复制执行的验证命令，以及应看到的结果。
4. 本地 Git commit、分支和工作区状态；提交失败时说明原因，不声称已提交。
5. 当前待确认问题；末尾询问是否通过本任务并允许进入下一任务。

在 README 维护一个小表即可：任务、状态、验证摘要、用户确认状态。不要新增复杂任务数据库。用户批准记录只能来自用户的明确回复，Codex 不得自行填为已通过。

## 8. 本地 Git 版本管理

用户仅使用本地 Git，不配置远端，不 push。每个可审阅的逻辑修改批次，包括文档、配置，都在验证后 commit；不要求每保存一行就提交。

开始先检查 status/diff，保留用户已有改动。验证后明确暂存本任务文件，不用 git add . 混入无关内容；提交并报告哈希及工作区状态。任务验收对应一个明确 commit。

不提交原始数据、缓存、outputs、大权重或凭据。仓库配置使用可移植路径或空值；个人绝对路径仅作本地未提交修改或命令行输入，实际运行配置保存在被忽略的 outputs 中。

不擅自 reset、重写历史或覆盖用户工作。缺少提交身份先询问，不编造。commit 失败时保留结果并说明；解决问题或获得用户例外授权前不进入下一任务。

两份交付文件由用户放入其本地项目，在 T0 一并提交；当前会话不替用户虚构本地机器的提交记录。

## 9. 参考材料与已知限制

- 工作区现有 `Proposed_Method_Revised.md`：用于核对负 ELBO、固定方差似然、排序目标、L2 和冻结训练；未指定具体神经网络层数。
- [参考预处理 Notebook](https://github.com/KBaja/UnsupervisedHI/blob/main/CMAPSS/01_data_preprocess.ipynb)
- [参考序列处理函数](https://github.com/KBaja/UnsupervisedHI/blob/main/CMAPSS/pre_process_utils.py)
- [参考模型 Notebook](https://github.com/KBaja/UnsupervisedHI/blob/main/CMAPSS/03_train_evaluate_proposals.ipynb)
- [参考评价函数](https://github.com/KBaja/UnsupervisedHI/blob/main/CMAPSS/eval_utils.py)

参考仓库的 TensorFlow 模型不是 OCALD。重用代码前核对许可，保留必要来源说明；若缺少明确许可，不默认整段复制授权。可根据已确认的指标定义独立实现。实现时记录使用的仓库 commit，避免日后 main 分支变化造成口径不明。

本方案已核对结构和任务依赖；尚未在真实 DS03 上执行任何训练或验证。用户的本地项目尚未在本会话打开；两份文档放入项目后，在 T0 纳入本地 Git。当前交付不宣称已在用户机器上提交。

## 10. 本地 Codex 启动指令

把本文件和 AGENTS.md 放入项目根目录，发送：

> 阅读 AGENTS.md 和 OCALD_Implementation_Plan.md，先执行 T0。T1–T4 的设计与数值配置已经批准，不重复询问；缺少实际路径或 Git 身份时再问我。只完成 T0，运行验证并提交本地 Git，提供验证命令、结果和 commit。等待我确认后再开始 T1。不配置远端，不 push。
