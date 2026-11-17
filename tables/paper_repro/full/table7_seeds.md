# Bang 7 qua nhieu bo du lieu - MSE trung binh 5 POS

Seed 0 = forecast.csv; cac seed khac = forecast_seeds.csv. Gia tri nho nhat moi dong in dam.

| Seed | RF | RBF | KELM | ELM | ARIMA | OUR | OUR-holdout | OUR-DE |
|---|---|---|---|---|---|---|---|---|
| 0 | **3.94** | 5.05 | 17.11 | 4.56 | 8.05 | 1.5e+07 | 5.92 | 6.31 |
| 1 | **4.52** | 5.45 | 22.93 | 5.90 | 7.66 | 5.7e+07 | 5.69 | 6.22 |
| 2 | 4.98 | **4.96** | 22.39 | 5.31 | 10.49 | 1.1e+07 | 7.48 | 5.09 |
| 3 | **4.87** | 7.39 | 23.61 | 7.52 | 9.97 | 5.2e+08 | 8.55 | 15.66 |
| 4 | 5.35 | **5.27** | 19.83 | 6.07 | 6.72 | 3.2e+11 | 5.87 | 9.39 |
| Trung vi | 4.87 | 5.27 | 22.39 | 5.90 | 8.05 | 5.7e+07 | 5.92 | 6.31 |

So seed moi mo hinh tot nhat: RF 3, RBF 2
So cap (seed, POS) moi mo hinh tot nhat (tren 25): RF 10, RBF 5, ELM 5, ARIMA 3, OUR-DE 2