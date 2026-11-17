# Kiem dinh thong ke (J_ref, 240 khoi ghep cap day du)
Friedman: chi2=462.37, p=6.81e-100

Rang trung binh (thap = tot): alns_dynamic=1.37, resolve=2.12, alns_static=2.69, greedy=3.82

Wilcoxon `alns_dynamic` vs con lai (hieu chinh Holm); A12>0.5: `alns_dynamic` co chi phi thap hon; Cliff delta<0: `alns_dynamic` tot hon.

| vs          |   mean_ref |   mean_other |   rel_gain_pct |    p_raw |   A12 |   cliffs_delta |   rank_biserial |   n |   p_holm | signif_0.05   |
|:------------|-----------:|-------------:|---------------:|---------:|------:|---------------:|----------------:|----:|---------:|:--------------|
| alns_static |   1.96e+03 |     2.53e+03 |           22.5 | 1.92e-32 | 0.605 |         -0.21  |           0.742 | 240 | 3.83e-32 | True          |
| greedy      |   1.96e+03 |     3.3e+03  |           40.6 | 1.78e-40 | 0.721 |         -0.443 |           0.967 | 240 | 5.33e-40 | True          |
| resolve     |   1.96e+03 |     2.21e+03 |           11.3 | 8.18e-23 | 0.557 |         -0.115 |           0.558 | 240 | 8.18e-23 | True          |
