# S2：排序梯度进入表示学习

本任务只新增 `ocald_s2.py` 与本说明，不改任何正式代码、`config.yaml`、既有 README 或
已有 outputs。用户已确认实现计划，采用 A：4 个设备×4 条 flight 的主 batch 方案。
S2 是独立诊断，不替换正式两阶段 OCALD。

## 已确认方法

```text
loss = representation_loss['loss'] + lambda * order_loss + 1e-4 * head_l2
r = HIHead(mu_I)
hi_raw = sigmoid(-r)
```

复用原 `OCALD` 结构和 `representation_loss`：X/mask 经 GRU 得到两块后验，W/mask
只决定 C 条件先验；decoder 只读采样 z_I/z_C 和时间位置。重构按 flight 内有效元素
求和，再与 KL_I/KL_C 相加并按 flight 平均，sigma_x_squared=1、KL 系数均为 1。
排序读取同一次 forward 的确定性 mu_I，不采样、不 detach、不标准化。
8→16→1 HI 头与三个表示模块共同训练，头 L2 含偏置，系数不随 lambda 改变。

从随机初始化运行 lambda=0.1/1/10、seed=42/43/44，共九次，均为 Adam lr=0.001、
weight_decay=0、100 epochs；全模型梯度范数共同裁剪到 5。保存最后一轮，无早停、
KL 退火、额外分支或额外损失，不加载旧 checkpoint 微调，也不追加正式 Stage 2。

| batch 项目 | 口径 |
|---|---|
| 设备 | 每 batch 均匀无放回抽 4 个不同 dev unit |
| flight | 每设备均匀无放回抽 4 条，组内无 gap≥5 对则重抽该组 |
| 跨 batch/epoch | 可重复抽样 |
| 排序对 | 每 batch 抽 16 对，允许重复；现有 sampler 先均匀选有效设备，再均匀选该设备有效对 |
| 候选范围 | 当前 batch 内；不等价于全 dev 所有对的均匀采样 |
| 每 epoch | 42 batch，共 672 次完整 flight 抽样及 672 对 |
| 配对方向 | 同 unit、cycle_late−cycle_early≥5，loss=softplus(r_early−r_late)，kappa=1 |

同 seed 的三个 lambda 共享初始化、完整 batch/配对序列及后验采样随机序列。batch RNG、
配对 RNG 与网络初始化/后验采样隔离；保留序列和各运行 RNG 最终哈希。新增分组抽样
与历史正式基线的普通 shuffle 不同，结果不能将所有差异唯一归因于排序项。
仅实现已选的 grouped 方案；随机 batch、无对时排序为 0 的备选未执行。

## 使用命令

在项目根目录、现有 Python 环境中执行，无新增依赖。CUDA 可用则用 CUDA，否则 CPU，
记录实际设备；不承诺跨硬件逐位一致。正式配置只读，实际 lambda/seed 记录在 S2 设置
和 checkpoint 中，不改正式配置的 seed=42 或两阶段语义。

```bash
python ocald_s2.py --config config.yaml --batch-scheme grouped --smoke
python ocald_s2.py --config config.yaml --batch-scheme grouped
python ocald_s2.py --config config.yaml --verify-only --run-dir outputs/ocald_s2/实际时间戳
```

完整入口固定运行获批的九组联合训练及 27 个探针，不自动额外跑 lambda=0 或其他预算。
每次新建 `outputs/ocald_s2/<UTC时间戳>/`，拒绝覆盖已有目录。
smoke 每设备最多选 8 条覆盖周期范围的完整 flight，dev/test=72/48；九组联合模型
各 3 epochs、每 epoch 2 batch，27 个探针各 3 epochs。数据归一化仍复用原 processed，
HI 缩放及探针标准化只用 smoke dev 子集拟合。smoke 不作为正式实验结论。

`--e1-results` 可显式传入 E1 的既有 `results_summary.csv`，默认指向已完成的
`outputs/baseline_ref/20261012T080807303596Z/results_summary.csv`。缺文件即报错，不自动
选择最新运行或填入替代数值；只读取比较结果，不重训 E1。

## 评价和探针

cycle/unit 仅用于抽样、配对、键关联和评价，不进入网络。全部联合训练及探针完成后才
读取 HI ground truth 与 RUL，按 `(unit, cycle)` 关联，不能参与训练或选择 lambda。

- 主模型排序：每设备穷举全部 gap≥5 对，以 r_late>r_early 为正确，平局计 0，设备等权。
  order 沿用原 r 的 softplus，有符号 HI/cycle Spearman 使用 raw HI，保留符号。
- 完整指标：`metrics.evaluate_metrics` 分别评价 raw/scaled 的 MAE/MAPE/M/趋势性/FC/MI。
  MI 统一 seed=42；scaled 复用 `scale_hi`，dev 全部最终 raw HI 拟合一个全局 min/max，
  test 复用、不裁剪、不翻转。Sigmoid 的 0/1 饱和计数显式报告。
- 零 dev 极差：scaled 不可用，保留 raw 和 r 排序；非有限指标记录原公式不可用原因，
  不加 epsilon、不修公式。种子汇总使用总体标准差 ddof=0，记录有效种子数。
- 清零分析：后验均值、I/C 同时清零、仅 I 清零、仅 C 清零；原 reconstruction_loss
  求和口径，flight 等权，另报告总 SSE/有效元素数的 pooled MSE，无采样。
- 探针：每个联合模型的 I、C、I+C 各训练一个独立 HIHead，共 27 个。完整运行各
  100 epochs，继承对应模型 seed；表示和联合 HI 头均冻结。dev-only ddof=0 标准化，
  test 复用；零标准差分母为 1 并记录。结构为 8/8/16→16→1，I/C 初始权重相同。
  沿用 D2：Adam lr=0.001、weight_decay=0、每轮 N_dev 对、batch=128 对、gap≥5、
  kappa=1、含偏置显式 L2=1e-4，保存最后一轮；同 seed 全部 lambda/探针共享配对。

## 产物

| 文件 | 内容 |
|---|---|
| run_settings.json | 实际设置、flight 键、设备、旧文件及 E1 CSV 哈希 |
| sampling_schedule.pt | 各种子的联合 batch/局部配对及探针全 dev 配对序列 |
| lambda_*/seed_*/checkpoint.pt | stage=ocald_s2 的完整模型、初始/最终参数、后验均值、r/raw 预测、dev HI 缩放、实际种子与 lambda |
| lambda_*/seed_*/losses.csv | 总损失、ELBO、重构、KL_I/KL_C、排序、未加权 L2；每 epoch 首 batch 的排序编码器梯度范数及裁剪前总梯度范数 |
| lambda_*/seed_*/predictions.csv / metrics.json | 主模型的 r/raw/scaled、HI/RUL 评价标签及完整双尺度指标 |
| lambda_*/seed_*/probes/I,C,IC/ | 每个探针的 head.pt、标准化参数、losses.csv、无标签预测表 |
| lambda_*/seed_42/training_curves.png | 重构/KL_I/KL_C/order 四面板训练曲线 |
| lambda_*/seed_42/hi_curves_dev.png / hi_curves_test.png | 完整设备 raw/scaled 曲线及参考 HI，共六张 |
| ranking_per_unit.csv | 主模型逐设备和总体排序、对数/平局数/正确数 |
| results_per_seed.csv / results_summary.csv | 主模型各 lambda/seed/split 的结果及三种子均值/总体标准差 |
| latent_ablation.csv | 九个模型、两个 split、四种设置的重构和元素 MSE |
| probe_results.csv / probe_summary.csv | 探针逐设备/总体排序和跨种子汇总 |
| comparison.csv | S2 与历史 OCALD、标准化 mu_I、F1、E1 V_C/V_ORD，含来源和种子数 |
| verification.json / summary.txt | 检查结果及简短说明 |

S2 checkpoint 有独立加载函数 `load_s2(path, device)`，不得将其当正式两阶段 `ocald.pt`
使用。已有文件、数据、weights 和 outputs 不进入本次本地 Git 提交。

## 验证与确认记录

关键检查包括：同设备早晚对方向；排序项单独反传到 GRU/mu_I 的非零有限梯度；排序项
不直接传入 decoder/条件先验；padding 改值和追加 padding 不改变确定性表示/HI；补零
不参与重构、mask 未变；ELBO 分项关系和联合 L2 权重；参数更新；共享初始化/RNG；
探针冻结；checkpoint/探针加载；指标、清零结果、排序 CSV 和种子汇总重算。
现有 `verify.py` 的七项另行运行，图与日志写新目录，成功打印 PASS，失败退出非零。

| 任务 | 状态 | 验证 | 用户确认 |
|---|---|---|---|
| S2 | 实现、smoke、九组完整联合训练与 27 个完整探针完成 | 保存后重验、独立 CSV/配对/标签/缩放检查及原七项 verify 均通过 | 用户已确认方案 A、目标/L2 口径和九次训练/27 探针预算；完成验收待确认 |

通过的 smoke 输出为 `outputs/ocald_s2/20261012T104352915387Z/`。首次检查在训练前
发现 cuDNN GRU 反向需要 train-mode 前向；修复后第二次完成短程训练，复核发现内存
预测列 float32 与 CSV 重读 float64 的 FC 数值不一致。评价现在将同一预测值以 float64
送入现有指标，保存与重读一致，不改预测、指标公式或训练损失。前两次未通过的独立
输出目录保留，均不作为通过记录；上述第三次完整 smoke 全部通过。

完整输出为 `outputs/ocald_s2/20261012T104802398175Z/`。九组联合模型均完成 100 epochs，
27 个探针均完成 100 epochs；dev/test 每模型完整评价 663/438 条 flight。
下表为三种子均值±总体标准差，正确率标准差单位为百分点。

| lambda | split | 排序正确率 (%) | 有符号 HI/cycle Spearman | 穷举设备等权 order |
|---|---|---|---|---|
| 0.1 | dev | 53.62 ± 1.07 | −0.094554 ± 0.024966 | 0.693001 |
| 0.1 | test | 51.38 ± 2.81 | −0.036855 ± 0.069557 | 0.693048 |
| 1 | dev | 54.89 ± 0.57 | −0.127662 ± 0.013543 | 0.688151 |
| 1 | test | 52.34 ± 0.43 | −0.055173 ± 0.010240 | 0.692480 |
| 10 | dev | 63.61 ± 9.68 | −0.339562 ± 0.232330 | 0.653258 |
| 10 | test | 62.97 ± 12.05 | −0.315292 ± 0.288719 | 0.652975 |

lambda=10 的 test 正确率按 seed=42/43/44 分别为 56.90%/79.80%/52.21%。平均改善主要
来自 seed=43，不能称为稳定解决失败。lambda=0.1/1 未显示明显测试排序改善。
全部 lambda 均报告，未按 test 选择模型或追加训练。

冻结表示探针的正确率如下，仍为三种子均值±总体标准差：

| lambda | 输入 | dev (%) | test (%) |
|---|---|---|---|
| 0.1 | I | 56.91 ± 0.59 | 53.11 ± 0.62 |
| 0.1 | C | 55.32 ± 0.13 | 52.68 ± 0.89 |
| 0.1 | I+C | 57.83 ± 1.10 | 53.63 ± 0.65 |
| 1 | I | 58.03 ± 1.10 | 54.13 ± 0.70 |
| 1 | C | 56.09 ± 1.37 | 53.09 ± 0.91 |
| 1 | I+C | 59.02 ± 0.82 | 54.22 ± 1.46 |
| 10 | I | 66.55 ± 10.30 | 63.96 ± 12.68 |
| 10 | C | 57.83 ± 2.40 | 54.72 ± 3.00 |
| 10 | I+C | 68.25 ± 10.56 | 66.25 ± 12.14 |

清零结果必须区分种子：seed=42/43 的 I 清零重构变化很小，而 seed=44 明显变大。
例如 lambda=10 时，dev/test 后验均值重构分别为：seed=42 的 26.2757/29.9516，
seed=43 的 27.2248/31.0450，seed=44 的 26.1086/29.6426；I 清零后分别为
26.3268/30.0177、27.1983/31.0056、68.9379/93.9952。全部模型的 C 清零误差均明显增大。
不能把所有种子统一描述为 I 没有重构贡献，也不能仅据清零结果认定物理解耦或后验塌缩。
四设置×两 split×九模型的 72 行及每元素 MSE 均在 `latent_ablation.csv`。

raw/scaled 的全部六项指标和总体标准差在 `results_summary.csv`，各指标均有三个有效
种子。当前所有预测均未达到 raw HI=0 或 1；部分 FC 因原公式中的平均首末下降量为负
而非常大，保留数值和原因，不裁剪，不解释为良好一致性。seed=42 raw 曲线仍大多较平；
Min–Max 放大其变化不意味着物理尺度标定。三张训练曲线和六张 HI 曲线均已查看，曲线点
与 CSV 一致，无平滑或测试集重缩放。

本次排序梯度路径确实存在：九组首个真实 batch 的未乘 lambda、未裁剪的编码器排序
梯度范数范围为 0.002625..0.015361；每 epoch 首 batch 的后续范数也记录在 losses.csv，
不假定后续所有梯度仍足够大。所有参数模块更新、同 seed 三档初始化/后验采样 RNG
一致、探针全模型冻结、保存后加载、预测及清零重算均通过。

原 `verify.py` 的 data/model/metrics/representation/hi/evaluation/data-plot 七项均通过，
日志位于 `outputs/ocald_s2/20261012T104620749636Z/verification_logs/`。representation
使用 `outputs/representation_formal_20261010T102117772658Z/representation.pt`，hi/evaluation
使用 `outputs/hi_formal_20261010T104138118262Z/`，data-plot 用临时配置仅将检查图写到
该新验证目录，没有更改正式配置或覆盖旧图。

另独立核对三套 100×42×16 flight 抽样和局部排序对、三套 100×663 探针配对；NumPy/
SciPy 重算全部有效对数/平局/正确率/softplus/有符号 Spearman、设备等权、ddof=0 汇总；
重新读取参考 HI/RUL 核对键关联，核对 dev-only 缩放、ELBO/L2 权重、masked MSE 元素数
和现有文件哈希。独立记录位于完整输出的 `verification_logs/independent_results.json`。
产物为 9 个联合 checkpoint、27 个探针 head、900 行联合训练日志、2700 行探针日志、
153 行主模型设备排序、459 行探针排序、18 行主模型逐种子汇总和 16 行来源标记对照。

在当前设置中，较大排序权重可在部分种子改善 I 的顺序信息，但改善不稳定，探针也未
接近 F1/E1 的约 94–95% 测试正确率。结果不支持仅保证排序梯度回传就能稳定解决问题；
由于分组抽样和历史基线不同，本次也不能唯一归因所有差异。未运行额外 lambda、随机
batch 备选、额外 Stage 2 或本任务之外的分析。
