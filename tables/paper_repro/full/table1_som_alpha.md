# Bang 1 - 3D-SOM voi alpha = 0 va alpha = 3 (100 task, A:B:C = 1:3:6)

Do 'thuan loai' = ti le task thuoc loai chiem da so cua moi UAV, trung binh tren cac UAV va cac lan chay.

## alpha = 0.0

| UAV | Bai bao A/B/C | Ta chay (lan 0) A/B/C |
|---|---|---|
| UAV1 | 3/9/18 | 2/5/11 |
| UAV2 | 2/2/10 | 0/3/13 |
| UAV3 | 1/6/9 | 3/7/10 |
| UAV4 | 2/5/16 | 3/8/15 |
| UAV5 | 2/8/7 | 2/7/11 |

Do thuan loai: bai bao 0.61, ta chay 0.60 (trung binh 10 lan).

So UAV khong nhan task nao (tung lan chay): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
So task toi da tren 1 UAV (tung lan chay): [26, 31, 26, 27, 26, 22, 26, 23, 25, 25]

## alpha = 3.0

| UAV | Bai bao A/B/C | Ta chay (lan 0) A/B/C |
|---|---|---|
| UAV1 | 4/12/0 | 0/0/60 |
| UAV2 | 6/12/0 | 0/0/0 |
| UAV3 | 0/0/18 | 0/30/0 |
| UAV4 | 0/6/18 | 0/0/0 |
| UAV5 | 0/0/24 | 10/0/0 |

Do thuan loai: bai bao 0.83, ta chay 1.00 (trung binh 10 lan).

So UAV khong nhan task nao (tung lan chay): [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
So task toi da tren 1 UAV (tung lan chay): [60, 60, 60, 60, 60, 60, 60, 60, 60, 60]
