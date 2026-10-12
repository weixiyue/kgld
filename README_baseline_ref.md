# 论文式参考基线

本任务仅新增 `baseline_ref.py` 与本说明文件，不改正式 OCALD 的任何代码、配置或已有
运行结果。用户已确认实现计划和主 batch 方案；结果验收需用户另行确认。

## 来源与范围

参考 Bajarunas et al., [Health index estimation through integration of general knowledge
with unsupervised learning](https://arxiv.org/html/2405.04990v1)，第 4、5 节，以及用户提供的
`baseline/03_train_evaluate_proposals.ipynb`。使用 PyTorch 独立实现指定诊断，不直接复制
Notebook，不将其称为论文/参考代码的完整复现。

Notebook 是 TensorFlow 实现，实际配置为四层 `[128,64,32,16]`、kernel=85、50 epochs，
NG 对潜变量做 L2 normalization 后对相邻样本计算周期归一化差分；C 在整个 batch
计算相关性。这些与用户指定的本次诊断不同。本次使用论文三层结构、20 epochs 和
下列明确获批的实现口径。论文趋势约束文字称 Spearman，但公式为 Pearson。

## 已确认设置

| 项目 | 本次口径 |
|---|---|
| 数据 | 只读取 `config.yaml` 指定的 `data/processed/ds03.pt`；dev unit 1–9、test unit 10–15 |
| 样本 | 完整 flight；实际全 dev/test 最大长度 2029，固定补零，无截断 |
| 编码器 | 只读 X；Conv1d 14→128→64→16，各层 ReLU，Flatten→Linear→标量 Z |
| 解码器 | 沿时间复制 Z，拼接 W(t)；Conv1d 5→16→64→128，各层 ReLU，逐时刻 Linear→14 |
| 卷积假设 | 所有层 kernel=11、stride=1、padding=5，无池化、归一化、dropout |
| 输出假设 | Z 和重构最后一层均线性，无 Sigmoid，无 KL/采样/latent normalization |
| 初始化 | PyTorch 默认初始化；同种子四变体初始参数逐值相同 |
| 重构 | 每 flight 有效元素 MAE，再 batch 等权平均；补零不计损失 |
| V0 | 仅 MAE，约束项为 0 |
| V_NG | 同 unit 全部 batch 内 gap≥5 对，mean relu(Z_late−Z_early)，不除周期差 |
| V_C | 每 unit 单独计算 Z/cycle Pearson，再设备等权平均，最小化使其趋向 −1 |
| V_ORD | 同 NG 的对，mean softplus(Z_late−Z_early)，温度 1 |
| 权重 | 三个约束变体 λ=1，联合回传编码器和重构路径；V0 λ=0 |
| 主 batch | 无放回选 4 个不同 dev unit，各无放回选 5 flight；无 gap≥5 对则重抽该组 |
| epoch | 正式 34 batch，680 次 flight 抽样；跨 batch/epoch 可重复 |
| 优化 | Adam lr=1e-4、weight_decay=0、其他参数默认；20 epochs；种子 42/43/44 |
| 选模 | 保存最后一轮；无早停、调度、额外正则或梯度裁剪 |
| C 数值假设 | Z 中心化 L2 norm 下限 1e-8；常数 Z 相关记 0，次数写入 losses.csv |
| HI | (Z−dev_min)/(dev_max−dev_min)；test 复用，不裁剪，不按结果翻转 |
| MI | 所有变体/种子统一采用现有函数的 seed=42 |

cycle/unit 只用于约束、抽样、键关联和评价，不作为网络输入。标签在所有变体/种子训练
全部结束后才读取。输入 Min–Max 不重新拟合。T_max 来自全数据形状，是固定补零长度，
不使用 test 数值拟合模型或归一化。卷积中间层不额外施加 mask，因此 Conv/Flatten
仍可能利用补零位置隐含的长度信息；“padding 不计损失”不保证模型不使用长度信息。

## 命令

在项目根目录及现有 Python 环境中运行，无新增依赖。自动选 CUDA，否则 CPU；记录
实际设备，不承诺跨设备逐位一致。

```bash
python baseline_ref.py --config config.yaml --smoke
python baseline_ref.py --config config.yaml --epochs 20 --lr 1e-4 --seeds 42 43 44

# 将路径替换为命令实际打印的完整结果目录：只读重验，不训练、不改输出。
python baseline_ref.py --config config.yaml --verify-only --run-dir outputs/baseline_ref/实际时间戳
```

`--epochs`、`--lr`、`--seeds` 可用于用户明确启动的灵敏度对照；本次只执行上述主设置。
smoke 使用每台设备最多 8 条覆盖周期范围的完整 flight，dev/test 为 72/48 条；固定
T_max=2029，四变体/三个种子各 3 epochs，每 epoch 2 batch。smoke 的 HI min/max 仅在
该训练子集拟合，不能当正式结果。正式运行从初始权重重新开始，不继承 smoke。

shape、X-only 信息路径、mask 排除及补零零梯度、同设备早晚对方向、三个约束的损失方向
和编码器梯度、常数 C 有限梯度、HI 方向等检查先于训练执行。成功打印 `PASS`；参数、
检查或训练失败退出非零，不为通过检查静默改变方法。

## 输出和评价

每次新建 `outputs/baseline_ref/<UTC时间戳>/`，`exist_ok=False`，不覆盖旧运行。
输出均被现有 `.gitignore` 排除，不提交数据、权重或 outputs。

| 输出 | 内容 |
|---|---|
| run_settings.json | 实际参数、假设、flight 键、设备、受保护现有文件和 processed 数据的哈希 |
| batch_schedule.pt | 各种子的完整 epoch/batch 抽样索引，四变体共享 |
| V*/seed_*/checkpoint.pt | 初始/最终参数、训练设置、归一化、dev HI min/max、最终 Z 预测 |
| V*/seed_*/losses.csv | 每轮 loss/mae/constraint、抽样数量、常数 Z 的 unit-batch 数 |
| V*/seed_*/predictions.csv | split/unit/cycle/Z/hi_scaled/hi_gt/RUL；按键关联，无预测平滑 |
| V*/seed_*/metrics.json | 现有 scaled HI 指标，整体与逐设备值、计数、不可用原因/警告 |
| V*/seed_42/hi_curves_dev.png / hi_curves_test.png | 所有 dev/test 设备曲线与参考 HI，标注现有指标 |
| ranking_per_unit.csv | 穷举有效对的逐设备和设备等权汇总，含平局/正确/总对数 |
| results_per_seed.csv | 每个变体、seed、split 的总体排序、指标及方向标记 |
| results_summary.csv | 跨种子均值/总体标准差 ddof=0，以及各指标有效种子数 |
| comparison.csv | 当前结果及用户指定的既有 OCALD、标准化 mu_I、F1 排序正确率 |
| verification.json | 初始化/抽样/保存后加载、指标重算及现有文件未变检查结果 |
| summary.txt | 设置、结果、方向标记和解释限制 |

排序穷举 gap≥5 的全部同设备对，以 Z_early>Z_late 为正确，平局计 0，设备等权平均。
有符号 HI/cycle Spearman 也按设备等权平均，保持符号；正相关标记为 rising，绝不人为
翻转。评价 order_hi=mean softplus(HI_late−HI_early)，这才与下降方向一致；同时记录
order_z 以区分 Min–Max 改变数值尺度的影响。不同模型的分数尺度不同，排序正确率和
有符号 Spearman 是主要可比项，order 数值不能直接视为相同模型能力尺度。

MAE/MAPE/M/趋势性/FC/MI 导入 `metrics.evaluate_metrics`，不改公式；MAPE 排除零 HI_gt。
零 dev 极差时 scaled 指标不可用，保留 Z 的排序/Spearman；非有限指标记录不可用原因，
不加 epsilon、不裁剪或伪造有效值。跨种子汇总记录有效种子数，缺失时不冒充三个种子
全部有效。dev 是训练集，不作为泛化证据。

对照表使用用户指定的历史正确率：OCALD 正式 54.35%/52.35%（单次）、标准化 mu_I
58.26%/55.99%（seed=42）、F1 93.49%/94.09%（五种子均值）。历史行标出来源、种子数，
不声称本次重训或与当前相同预算；不为历史缺失指标填自选数值。

## 执行与验收记录

| 任务 | 状态 | 验证 | 用户确认 |
|---|---|---|---|
| 参考基线 | 实现、smoke、20 epochs 主设置及结果复核完成 | CUDA 12 模型 smoke/完整运行通过；现有七项 verify 通过；CSV 独立复核通过 | 实现计划及主 batch 方案已明确确认；完成验收待确认 |

本次 CUDA smoke 输出为 `outputs/baseline_ref/20261012T080241023526Z/`，完整运行输出为
`outputs/baseline_ref/20261012T080807303596Z/`。12 个模型均使用 seed=42/43/44、lr=1e-4、
20 epochs、34 batch/epoch，保存最后一轮；全部指标均有三个有效种子。

| 变体 | split | 排序正确率 (%)，均值±总体标准差 | 有符号 Spearman，均值±总体标准差 | order_hi 均值 |
|---|---|---|---|---|
| V0 | dev | 46.95 ± 7.86 | +0.078779 ± 0.204255 | 0.701513 |
| V0 | test | 46.37 ± 7.27 | +0.095879 ± 0.186518 | 0.706371 |
| V_NG | dev | 89.90 ± 7.59 | −0.868245 ± 0.121210 | 0.642064 |
| V_NG | test | 85.87 ± 8.88 | −0.801382 ± 0.154695 | 0.640776 |
| V_C | dev | 95.84 ± 1.27 | −0.961586 ± 0.013964 | 0.586661 |
| V_C | test | 94.23 ± 0.57 | −0.944254 ± 0.005669 | 0.585546 |
| V_ORD | dev | 96.34 ± 0.99 | −0.965758 ± 0.011226 | 0.583794 |
| V_ORD | test | 95.45 ± 0.73 | −0.957824 ± 0.009046 | 0.580667 |

现有 scaled HI 指标的三种子均值如下；完整总体标准差均在 `results_summary.csv`。

| 变体 | split | MAE | MAPE (%) | M | 趋势性 | FC | MI |
|---|---|---|---|---|---|---|---|
| V0 | dev | 0.338840 | 74.677430 | 0.048070 | 0.218723 | 1108.621548 | 0.055121 |
| V0 | test | 0.322565 | 79.452790 | 0.063183 | 0.207368 | 2.636163 | 0.086490 |
| V_NG | dev | 0.326729 | 55.163146 | 0.181893 | 0.868245 | 0.728991 | 0.624375 |
| V_NG | test | 0.303057 | 53.007430 | 0.120391 | 0.801382 | 0.757848 | 0.552560 |
| V_C | dev | 0.057486 | 14.559787 | 0.198197 | 0.961586 | 0.906790 | 0.766913 |
| V_C | test | 0.053921 | 12.499920 | 0.200423 | 0.944254 | 0.924585 | 0.733502 |
| V_ORD | dev | 0.042053 | 14.566240 | 0.261274 | 0.965758 | 0.907425 | 0.782053 |
| V_ORD | test | 0.041595 | 12.657850 | 0.246566 | 0.957824 | 0.915544 | 0.763225 |

V0 的 seed=43、44 在 dev/test 的总体有符号 Spearman 都为正，已标记 rising，未翻转。
V0 的 FC 因部分种子的首末 HI 方向相反而超过 1，保留现有公式结果及原因；不能将该数值
解释为良好一致性。三个约束变体均为下降方向。V_NG 种子间差异较大，不能只用 seed=42
曲线代表其三种子平均表现。12 个模型没有遇到常数 Z unit-batch。

复用同一 processed 数据后，带 C/ORD 约束的参考 AE 能在 test 得到约 94–95% 的排序
正确率，而无约束 V0 整体较弱。这支持该管线中存在可利用的顺序信号，OCALD 的失败
更应从表示/模型/训练设计排查；本次并不能唯一定位原因或证明数据管线不存在任何问题。
差异还包括网络容量、确定性表示、W 直接进入解码器及联合约束训练，不能单独归因某一项。
固定补零长度和卷积/Flatten 的长度信息限制仍适用，不声称因果解耦或完整论文复现。

现有 `verify.py` 的 data/model/metrics/representation/hi/evaluation/data-plot 七项均通过，
日志位于 `outputs/baseline_ref/20261012T080437639220Z/verification_logs/`。representation
使用 `outputs/representation_formal_20261010T102117772658Z/representation.pt`，hi/evaluation
使用 `outputs/hi_formal_20261010T104138118262Z/`，data-plot 用该验证目录中的临时配置
仅改变输出根目录；未修改正式 config 或覆盖旧图。非法 epochs/lr、缺少 verify-only 目录
也核对了非零退出码。

完整运行保存 12 个 checkpoint、240 行训练日志、每模型 1101 条预测、204 行设备排序
记录、24 行逐种子结果、8 行种子汇总，以及 seed=42 的 8 张完整设备曲线，均已查看。
独立 NumPy/SciPy 重算有效对、平局、softplus、有符号 Spearman、设备等权和 ddof=0
种子汇总；另核对 20×34×20 的各种子抽样序列、四组共享初始化、loss=MAE+约束、
dev-only HI 参数和源文件/processed 数据哈希。保存后加载在 CUDA 上预测逐值一致。
独立复核记录位于完整运行的 `verification_logs/independent_results.json`。
最终 `--verify-only` 入口也已在完整运行上执行通过；它只读核对 checkpoint、CSV 预测、
metrics.json、逐设备/逐种子排序表、种子汇总及抽样序列，不改写 verification.json 或其他产物。
未执行随机 batch 备选、100 epochs/lr=1e-3 灵敏度实验或本任务之外的分析。
