# E3：编码器隔离诊断（T1/T2）

当前授权范围：只运行 T1、T2，seed=42。新建 `ocald_e3.py` 与本文；既有 E3 文件不存在，
没有删除 T3/T4 或其他种子实现。本版不实现、不运行 T3/T4。独立诊断不替换正式流程。

## 设置

- 复用 DS03 processed 数据与原归一化：完整 dev/test=663/438，unit=1–9/10–15。
- T1：导入原 SensorEncoder，取 mu_I；原有 C/方差输出头冻结，不参与目标。
- T2：Conv1d 14→128→64→16，kernel=11、padding=5、ReLU；输入与每层激活按 mask
  置零，按通道/时间展平，Linear→8。固定长度取 processed 的最大 flight 长度2029，
  仅使用形状信息，不拟合 test 数值；追加的无效尾部忽略，有效长度超过固定长度报错。
- 两者只有编码器与8→16→1 HIHead，r=head(mu_I)，HI=sigmoid(-r)。不实例化解码器/
  条件先验；不计算 ELBO/KL，不采样，不输入 W、unit、cycle、HI 或 RUL。
- loss=softplus(r_early−r_late).mean()+1e-4*head_l2，L2含偏置。
- Adam lr=.001、weight_decay=0、100 epochs，全模型梯度范数裁剪5，最后轮权重；
  无早停、调度、测试选模。PyTorch默认初始化，不增归一化、平滑或后处理。
- 直接复用 S2 joint_schedule：每batch无放回选4设备×4flight，无gap≥5对则重抽该组；
  跨batch允许重复。每batch均匀选设备再均匀选有效对，抽16对允许重复。
  每轮42batch/672次flight抽样。同seed两种架构共享batch/配对序列。
- 每epoch记录全部batch的未加L2的排序梯度、总梯度裁剪前后范数均值，同时记录骨干/
  mu_I头排序梯度和零范数batch比例。不同架构分别初始化，不声称同一初始化。
- 第1轮及每10轮评价dev穷举排序，100轮值对应最终模型。训练日志order为采样batch均值，
  最终评价order为穷举设备等权，两者分别标明。

## 评价与探针

复用 diagnostics_features.ranking_rows：同设备gap≥5全部对，以r_late>r_early为正确，
平局计0，设备等权；有符号Spearman基于sigmoid(-r)，不取绝对值、不翻转。
保存逐设备及总体dev/test表。最终raw HI缩放只使用dev极值，test复用、不裁剪。
为与S2保持评价一致，metrics.json另保留raw/scaled六项指标；不额外分析这些指标。

每个主模型仅训练一个标准化mu_I探针：8→16→1，dev均值/总体标准差，零标准差分母1并
记录，test复用不裁剪；冻结整个主模型。Adam lr=.001、weight_decay=0、100epochs，
每轮N_dev对、128对/batch，uniform_unit_then_uniform_valid_pair，gap≥5，kappa=1，
含偏置头L2=1e-4，最后轮权重，与D2一致。同seed两种探针共享完整配对序列。
HI/RUL在所有主训练及探针完成后才读取，仅用于评价与参考曲线。

## 命令

```bash
python ocald_e3.py --config config.yaml --variants T1 T2 --seeds 42 --smoke
python ocald_e3.py --config config.yaml --variants T1 T2 --seeds 42
python ocald_e3.py --verify-only --run-dir outputs/ocald_e3/实际时间戳
```

CLI可限定T1/T2和种子；本次仅执行42。`--smoke`选每设备最多8条完整flight，dev/test=72/48，
主模型和探针各3epochs，每轮2个主batch；卷积固定长度仍用完整数据最大长度。
CUDA可用则使用，否则CPU；重验应与产物使用相同设备，不保证跨硬件逐位一致。
默认只读对照CSV为 `outputs/ocald_s2/20261012T104802398175Z/results_summary.csv`，可用
`--s2-results`传入；缺文件报错，无自动回退。

## 输出与检查

每次新建outputs/ocald_e3/UTC时间戳，拒绝覆盖。根目录保存run_settings.json、
sampling_schedule.pt、results.csv、ranking_per_unit.csv、probe_results.csv、comparison.csv、
verification.json、summary.txt。配置/seed子目录保存checkpoint.pt、losses.csv、predictions.csv、
metrics.json、training_curves.png、hi_curves_dev/test.png、verification.json、probe_I/产物。
对照标明单种子/三种子均值/五种子均值和来源，不将不同种子数混称。

检查真实排序梯度进入骨干与mu_I、padding改值/追加不影响输出、mask不变、配对方向、
未用输出头冻结、主模型探针冻结、保存后预测/探针/排序/日志/指标重算和既有文件哈希。
原verify.py七项另运行；data-plot使用新目录内的临时配置，正式config.yaml保持原样。
验证失败退出非零；成功打印PASS。

| 任务 | 状态 | 用户确认 |
|---|---|---|
| E3 T1/T2 seed42 | CPU smoke、完整CUDA训练与两个完整探针、独立重验及原七项verify均通过 | 用户直接授权实现与运行；完成验收待确认 |

## 实际结果

完整输出：`outputs/ocald_e3/20261012T134553977143Z/`，RTX3090，主模型与探针均100epochs。
smoke输出：`outputs/ocald_e3/20261012T133953347223Z/`，CPU，各3epochs。
两种配置都只运行seed42，不计算跨种子均值或标准差，不运行T3/T4。

| 配置 | dev排序正确率 | test排序正确率 | dev/test有符号Spearman | 训练order首轮→末轮 | mu_I探针dev/test |
|---|---|---|---|---|---|
| T1 GRU | 94.1760% | 94.3528% | −0.930358 / −0.927117 | 0.693135→0.100589 | 94.1695% / 94.3417% |
| T2 Conv | 98.5961% | 97.1218% | 0 / 0（常数HI） | 0.700817→0.082984 | 97.5982% / 96.5607% |

两个训练order总体下降，均有轮间波动。每epoch平均排序编码器梯度范数，T1范围
0.006264..49.4719、末轮7.26046；T2范围0.256677..44.1047、末轮8.87125。
全部batch有限且非零；部分裁剪前梯度较大，已按全模型范数5裁剪，不存在非有限梯度。
不将单种子表现描述为多种子稳定性结论。

最终T1 raw HI极小（0..1.09215e-24），T2全部raw HI=0。排序正确率始终根据r，
Sigmoid饱和不改变该口径。T2常数HI的Spearman沿用metrics.py返回0的行为，
不声称其相关系数在数学上有定义；dev极差零，scaled指标/曲线不可用。
保留原预测和不可用标记，不翻转、不增加epsilon、不调整温度。

原verify.py的data/model/metrics/representation/hi/evaluation/data-plot七项通过；
日志位于完整输出下verification_logs/。representation验证使用20261010正式表示，
hi/evaluation使用20261010正式HI模型；data-plot仅通过临时配置输出到新目录。
独立--verify-only已重算主模型/探针表示、排序、日志、指标与缩放并通过。
CPU重验固定与训练相同的单线程设置，避免改变线程数造成浮点积累差异。

最简单的实现选择：T1原编码器保留并冻结未用C/方差头；T2只建立mu_I输出头；
卷积固定长度取processed形状最大值，默认初始化；无额外分支、损失或后处理。
