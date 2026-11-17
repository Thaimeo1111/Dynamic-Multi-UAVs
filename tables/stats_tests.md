# Kiem dinh thong ke (J_ref, 240 khoi ghep cap day du)
Friedman: chi2=472.28, p=4.85e-102

Rang trung binh (thap = tot): alns_dynamic=1.34, resolve=2.10, alns_static=2.75, greedy=3.81

Wilcoxon `alns_dynamic` vs con lai (hieu chinh Holm); A12>0.5: `alns_dynamic` co chi phi thap hon; Cliff delta<0: `alns_dynamic` tot hon.

| vs          |   mean_ref |   mean_other |   rel_gain_pct |    p_raw |   A12 |   cliffs_delta |   rank_biserial |   n |   p_holm | signif_0.05   |
|:------------|-----------:|-------------:|---------------:|---------:|------:|---------------:|----------------:|----:|---------:|:--------------|
| alns_static |    2.2e+03 |     2.95e+03 |           25.3 | 9.93e-35 | 0.6   |         -0.2   |           0.808 | 240 | 1.99e-34 | True          |
| greedy      |    2.2e+03 |     3.98e+03 |           44.7 | 2.06e-40 | 0.703 |         -0.405 |           0.95  | 240 | 6.18e-40 | True          |
| resolve     |    2.2e+03 |     2.47e+03 |           11   | 6.92e-23 | 0.554 |         -0.108 |           0.571 | 240 | 6.92e-23 | True          |
