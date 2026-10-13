# 取消KL_I：单种子联合消融

只运行seed42、lambda10。原OCALD网络、重构求和口径、sigma_x_squared=1、KL_C权重1、
I/C重参数化、同次前向mu_I排序保持S2规则；KL_I计算并记录，但不进入目标图。
目标为reconstruction+KL_C+10*order+1e-4*含偏置head_L2，不称为标准负ELBO。
Adam lr=.001、weight_decay=0、100联合epochs、全模型梯度裁剪5、最后轮权重。
无探针、排序正确率、额外种子/权重、早停、KL退火、温度修改或平滑。

直接导入S2的分组抽样：4个不同设备×每设备4条不同flight；无gap≥5对则重抽设备组，
跨batch可重复。每batch抽16对，先均匀设备再均匀有效对，允许重复；每轮42batch/672flight。
同seed42初始化与S2严格核对；完整batch/配对序列核对已有sampling_schedule.pt。
训练只用原torch.randn_like采样，最终后验采样RNG摘要与S2核对，不在训练中另作随机评价。
checkpoint stage=ocald_no_kli，含实际训练设置、列名、归一化、flight键、dev raw极值与预测，
不伪装为正式hi阶段或两阶段100/100。

## 使用

```bash
python ocald_no_kli.py --config config.yaml --smoke
python ocald_no_kli.py --config config.yaml
python ocald_no_kli.py --verify-only --run-dir outputs/ocald_no_kli/实际时间戳
```

--smoke沿用S2：每设备最多8条完整flight，dev/test=72/48，每轮2batch，共3epochs。
完整训练dev/test=663/438；processed及正式config只读，不改训练归一化。
默认只读S2来源为outputs/ocald_s2/20261012T104802398175Z/lambda_10/seed_42/checkpoint.pt，
可用--s2-checkpoint明确指定；缺失来源会注明，绝不编造或重训对照。
新建outputs/ocald_no_kli/UTC时间戳，不覆盖已有目录。

## 评价

复用evaluate.predict_table/attach_references/metric_report/plot_curves/evaluate，原metrics.py
六项公式及汇总不变。HI/RUL在最后训练更新后才读取，只做评价与参考曲线。
hi_raw=sigmoid(-r)，hi_scaled由全部dev最终raw预测全局min/max拟合，test复用、不裁剪。
诊断端点0/1允许但报告数量；dev极差零时scaled不可用；原公式非有限值记录null及原因/
原警告，不加epsilon。单调性与趋势性保留原公式，包括常数Spearman返回0的约定。
诊断报告与图标题记joint_epoch，不写成正式Stage1/Stage2预算。
正式checkpoint的加载/检查/报告不变。evaluate.py新增--output-dir，将S2重评到新的
s2_reference/，不写回源目录。

输出checkpoint.pt、losses.csv、actual_config.yaml、run_settings.json、sampling_schedule.pt、
mechanism_checks.json、predictions.csv、metrics.json、hi_curves.png、verification.json、
comparison.csv和summary.txt。comparison按dev/test与raw/scaled分开列示六项值及不可用原因。
S2的重评预测/指标/曲线单独保存在s2_reference/。

检查目标组成、KL_I目标梯度不存在、KL_C目标梯度1且进入条件先验、排序进入GRU/mu_I、
mask/配对方向、初始化/完整抽样/采样随机摘要、最后权重重载、HI/RUL键关联和dev-only缩放、
指标重算、无正确率/探针产物及共用代码未变。原训练不重跑；相关原verify接口另验证。
E3已完整归档并逐文件核对哈希，历史限制见项目README与outputs/diagnostic_archive/e3/。

## 实际运行结果与限制

完整运行：`outputs/ocald_no_kli/20261013T051448074601Z/`，CUDA RTX3090，seed42、lambda10，
100联合epochs；smoke：`outputs/ocald_no_kli/20261013T051209734691Z/`，CPU，3epochs。
完整新旧评价均为dev663/test438，原S2只重评、不重训。

下表为原S2→取消KL_I；MAPE单位%，其余沿用原指标口径：

| split/scale | MAE | MAPE | 单调性M | 趋势性 | FC | MI |
|---|---|---|---|---|---|---|
| dev/raw | 0.23889→0.39687 | 72.643→72.070 | 0.06181→0.04151 | 0.21521→0.10995 | 0.007552→4.30091e7 | 0.03161→0.03536 |
| dev/scaled | 0.41924→0.47047 | 75.776→77.979 | 0.06181→0.04151 | 0.21521→0.10995 | 0.007552→4.30090e7 | 0.03161→0.03536 |
| test/raw | 0.22298→0.37204 | 75.169→71.515 | 0.03096→0.03837 | 0.17528→0.05818 | 0.117097→1.77449e10 | 0.06535→0.03264 |
| test/scaled | 0.35475→0.41330 | 73.834→74.349 | 0.03096→0.03837 | 0.17528→0.05818 | 0.117097→1.77451e10 | 0.06535→0.03264 |

两模型两个split的raw=0/1饱和数量均0。新dev raw范围0.332563..0.548701，
scaled可用；S2 test scaled最大1.001365，保留、不裁剪。新FC远大于1，是原公式的
平均首末下降量为负所致，不应解读为改善；指标仍有限，完整运行无非有限指标。

末轮loss=27.587266、reconstruction=20.403491、KL_I=42.070666（仅记录）、KL_C=0.249267、
order=0.693406、未加权head_L2=4.481173。相同初始化SHA256、4200个batch与配对序列、
最终后验采样RNG摘要均与S2 lambda10/seed42一致。

新目标组成/KL梯度、GRU/mu_I排序路径、mask/方向、最后权重重载、参考键关联、
dev-only缩放、指标及对照表重算、无正确率/探针产物全部通过。相关原verify.py
model/metrics/representation/hi/evaluation五项通过，正式checkpoint原端点拒绝行为、
诊断端点与零极差/非有限JSON处理另核验通过。日志位于完整目录validation/。
未重跑既有训练或预处理，没有新增种子、lambda、探针或清零分析。
E3归档71文件逐一核验，原运行及代码删除，活动代码/文档无失效E3路径引用。

该单种子配对消融不支持“取消KL_I即可改善退化表示学习”：MAE与趋势性变差，test MI降低，
训练order仍约0.693。它不能排除KL_I在其他设置中的影响，也不能确定唯一失败根因或证明
物理解耦。本次完成验收待用户确认，不自动开展后续实验。
