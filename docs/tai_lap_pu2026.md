# Tái lập thí nghiệm Pu, Yang, Li (2026) - AST 172:111711

Bài: "Dynamic task allocation for multi-UAVs in disaster scenarios: a novel two-level
optimization method with forecasting" (`docs/1.pdf`). Bài viết bằng MATLAB và không công bố mã.
Bản tái lập ở đây là bản độc lập, viết bằng Python 3.11, nằm trong thư mục `paper_repro/`.

> **Trạng thái (2026-11-17):** đã có Bảng 1, Bảng 2, Bảng 7 và mô phỏng động.
> Bảng 4 (một loại task) và Bảng 5 (nhiều loại task) đang chạy, mục 4 sẽ bổ sung sau.
> Giải thích thuật ngữ (SOM, đấu giá, 2-Opt, ELM/KELM, AOO, ...): `docs/giai_thich_thuat_ngu.md`, mục 12.

## 0. Cách chạy và nơi để kết quả

```
py -3.11 -m paper_repro.run_paper_repro --exp table1 table2 forecast dynamic --full
py -3.11 -m paper_repro.forecast_seeds --seeds 1 2 3 4                           # Bảng 7 trên 4 bộ dữ liệu thêm (mục 3.2)
py -3.11 -m paper_repro.run_paper_repro --exp single multi --full --workers 4   # Bảng 4, 5 (vài giờ; chạy tiếp được)
py -3.11 -m paper_repro.report full                                              # sinh lại toàn bộ bảng + hình
py -3.11 -m pytest tests/test_paper_repro.py -q
```
Mọi con số trong báo cáo này đều lấy từ các bảng do `paper_repro.report` sinh ra.
Riêng mục 3.2 lấy từ `tables/paper_repro/full/table7_seeds.md`.

| Thí nghiệm | CSV thô | Bảng so sánh | Hình |
|---|---|---|---|
| Bảng 1 | `results/paper_repro/full/table1_som_alpha.csv` | `tables/paper_repro/full/table1_som_alpha.md` | |
| Bảng 2 | `results/paper_repro/full/table2_auction.csv` | `tables/paper_repro/full/table2_auction.md` | `figures/paper_repro/full/paths_100.png` |
| Bảng 7 | `results/paper_repro/full/forecast.csv`, `forecast_seeds.csv` | `tables/paper_repro/full/table7_forecast.md` | `figures/paper_repro/full/forecast_fit.png` |
| Hình 14 | `results/paper_repro/full/dynamic_events.csv` | `tables/paper_repro/full/dynamic_events.md` | `figures/paper_repro/full/dynamic_frames.png` |

### Thiết lập chung (theo bài, phần nào bài không nói rõ thì ghi là "ta chọn")
- Bản đồ 100×100, base đặt tại [50, 50], 5 UAV, task rải ngẫu nhiên đều trên bản đồ.
- Hàm chi phí Eq.14: `C = 1·P + 1.5·D + 2·R`. P là tổng quãng đường bay, D là độ lệch chuẩn quãng đường giữa các UAV, R là độ lệch chuẩn tổng điểm task giữa các UAV.
- **Độ lệch chuẩn:** Eq.16 của bài chia cho u, nhưng ví dụ số ở Hình 4 (std(20,15) = 3.533) lại là độ lệch chuẩn mẫu, tức chia cho u−1 như hàm `std` mặc định của MATLAB. Ta dùng u−1 để khớp với số của bài.
- TSP: chạy NN rồi 2-Opt, đường bay mở (không quay về base, theo §3.3.2). Với nhiều loại task thì xếp theo tầng: bay hết loại A, rồi B, rồi C.
- Nhiều loại task: tỉ lệ A:B:C = 1:3:6, điểm tương ứng 5/3/1, A có ưu tiên cao nhất.
- Bài không nêu số vòng lặp của SOM: ta chọn 100. Tốc độ học giảm từ 0.1 xuống 0.01, bán kính lân cận giảm từ 2 xuống 0.1 (Bảng 3).
- Algorithm 2 (đấu giá) có nhánh `else break`. Nếu cài đúng như vậy, vòng lặp dừng ngay khi gặp c′ = c. Ta cài theo ý của lời văn, tức thử chuyển sang mọi c′ ≠ c, tối đa 20 vòng.

---

## 1. Bảng 1 - 3D-SOM với α = 0 và α = 3

100 task, A:B:C = 1:3:6, 10 lần chạy (seed 0-9).
"Độ thuần loại" = tỉ lệ task thuộc loại chiếm đa số trong tập task của mỗi UAV,
tính trung bình trên các UAV có nhận task.

| | Bài: độ thuần | Ta: độ thuần (TB 10 lần) | Ta: số UAV không nhận task nào | Ta: số task tối đa trên 1 UAV |
|---|---|---|---|---|
| α = 0 | 0.61 | 0.60 | 0 / 5 ở cả 10 lần | 22-31 |
| α = 3 | 0.83 | **1.00** | **2 / 5 ở cả 10 lần** | **60** (cả 10 lần) |

Phân bổ ở lần chạy 0 (A/B/C):

| UAV | Bài α=0 | Ta α=0 | Bài α=3 | Ta α=3 |
|---|---|---|---|---|
| UAV1 | 3/9/18 | 2/5/11 | 4/12/0 | 0/0/60 |
| UAV2 | 2/2/10 | 0/3/13 | 6/12/0 | 0/0/0 |
| UAV3 | 1/6/9 | 3/7/10 | 0/0/18 | 0/30/0 |
| UAV4 | 2/5/16 | 3/8/15 | 0/6/18 | 0/0/0 |
| UAV5 | 2/8/7 | 2/7/11 | 0/0/24 | 10/0/0 |

**Nhận xét**
- **α = 0 khớp với bài:** độ thuần 0.60 so với 0.61. Khi α = 0, SOM chỉ gom theo vị trí, nên loại task trong mỗi cụm gần với tỉ lệ chung 1:3:6.
- **α = 3 lệch hẳn về định tính.** Ở bản của ta, khoảng cách theo loại lấn át khoảng cách vị trí. Cả 3 loại dồn vào đúng 3 neuron: một UAV ôm 60 task loại C, còn 2 UAV không nhận task nào (hiện tượng "neuron chết"). Chuyện này xảy ra ở cả 10 lần chạy, không phải do một seed xui.
- Bài vẫn dùng đủ 5 UAV, mỗi UAV 14-24 task, độ thuần 0.83. Như vậy trong cài đặt của bài, ảnh hưởng của α yếu hơn của ta. Có mấy cách giải thích:
  (a) bài chuẩn hóa thành phần loại theo cách khác, ví dụ loại ∈ {1, 2, 3} không chia lại về [0, 1];
  (b) α nhân vào bình phương khoảng cách thay vì khoảng cách;
  (c) bài có thêm một bước cân bằng tải mà không mô tả.
  Bài không cho đủ chi tiết để biết cách nào đúng.
- **Hệ quả:** hiệu quả của "3D" trong 3D-SOM rất nhạy với cách cài đặt chi tiết này. Ở bản của ta, khâu đấu giá phía sau phải gánh việc cân bằng lại tải. Bảng 5 sẽ cho biết OUR có còn giữ được lợi thế hay không.

## 2. Bảng 2 - Trước và sau tối ưu đấu giá (5 UAV, một loại task)

Mỗi quy mô có 10 kịch bản, phân bổ ban đầu bằng SOM với α = 0.
Theo bài, `Sum = Length + 3·Std` (cột Sum của bài khớp đúng công thức này), và cột cải thiện tính trên Sum.

| n | Bài: Sum trước → sau | Bài: cải thiện | Ta: Sum trước → sau | Ta: cải thiện | Ta: cải thiện theo Eq.14 | Ta: số vòng / thời gian (s) |
|---|---|---|---|---|---|---|
| 20 | 748.9 → 641.6 | 18.25% | 439.2 → 407.6 | 7.30% | 4.37% | 3.1 / 0.08 |
| 60 | 1019.0 → 864.1 | 15.20% | 729.1 → 690.3 | 5.26% | 3.58% | 3.1 / 0.32 |
| 100 | 1075.8 → 1034.9 | 3.80% | 905.7 → 861.7 | 4.78% | 3.77% | 4.6 / 0.95 |
| 150 | 1513.6 → 1201.0 | 20.65% | 1127.8 → 1047.0 | 7.10% | 5.35% | 4.5 / 1.73 |
| 200 | 1725.1 → 1439.7 | 16.55% | 1254.9 → 1209.8 | 3.53% | 2.58% | 5.8 / 3.32 |
| 250 | 1647.2 → 1511.7 | 8.23% | 1404.3 → 1340.2 | 4.57% | 3.32% | 5.0 / 4.08 |
| 300 | 1708.0 → 1500.0 | 12.18% | 1556.3 → 1471.5 | 5.37% | 3.78% | 6.7 / 7.67 |
| 350 | 1771.3 → 1660.0 | 6.28% | 1671.4 → 1587.7 | 4.99% | 3.52% | 6.3 / 10.14 |
| 400 | 1983.7 → 1724.9 | 13.05% | 1787.5 → 1682.8 | 5.84% | 4.27% | 5.6 / 11.21 |
| **TB** | | **12.69%** | | **5.41%** | | |

Bảng đầy đủ (Length và Std riêng): `tables/paper_repro/full/table2_auction.md`.

**Nhận xét**
- **Chiều hướng khớp với bài:** đấu giá luôn giảm chi phí. Std giảm ở 94% số lần chạy (85/90), còn Length tăng ở 1/3 số lần chạy. Nghĩa là đấu giá chấp nhận bay xa hơn một chút để chia tải đều hơn, giống bài (ví dụ n = 20 và n = 250 của bài cũng có Length tăng).
- **Mức cải thiện của ta chỉ khoảng 40% của bài** (5.4% so với 12.7%), và dao động theo n ít hơn hẳn (3.5-7.3% so với 3.8-20.7%). Hai nguyên nhân có thể:
  1. **Điểm xuất phát.** Std ban đầu của ta khoảng 25-52, của bài khoảng 20-104. SOM của bài để lại nhiều mất cân bằng hơn, nên đấu giá có nhiều chỗ để cải thiện hơn.
  2. **Mục tiêu tối ưu khác thước đo báo cáo.** Đấu giá tối ưu Eq.14 (trọng số Std là 1.5), nhưng bảng chấm theo Sum (trọng số Std là 3). Std sau đấu giá của bài rất nhỏ (1.4-27.8, có lần còn 1.4 ở n = 250), gợi ý rằng bài thực ra tối ưu một hàm phạt Std nặng hơn Eq.14. Đây là suy đoán, bài không nói rõ.
- **Length tuyệt đối của ta ngắn hơn bài 6-36%,** chênh nhiều nhất ở n nhỏ (n = 20: 365 so với 573). Bài không nói rõ Length có tính chặng từ base ra task đầu tiên, có quay về base, hay có tính chặng sạc. Ta tính đường mở từ base. Vì vậy chỉ nên so **% cải thiện**, không so giá trị tuyệt đối.
- Thời gian: 0.08-11 s bằng Python thuần, cùng bậc với bài.

## 3. Bảng 7 - Dự báo cung-cầu vật tư

**Dữ liệu (Eq.19-25):** 5 điểm cứu trợ (POS), mỗi điểm 100 bước, nhiễu `2·N(0,1)`.
Mô hình dùng 3 bước trước để dự báo bước kế tiếp. 80% đầu chuỗi để train (77 mẫu), 20% cuối để kiểm định (**chỉ 20 điểm**).
Ta chọn chu kỳ xung của F4 là 3, vì bài ghi U = "số bước tối đa", nếu hiểu đúng như vậy thì cả chuỗi chỉ có 1 xung.

**Các biến thể của mô hình đề xuất (RBF-MKELM):**
- `OUR`: dò 7 siêu tham số bằng AOO, mục tiêu là RMSE trên chính tập train, **đúng như bài** (Eq.40).
- `OUR-holdout`: vẫn dùng AOO, nhưng mục tiêu là RMSE trên 20% cuối của tập train. Tập kiểm định không được dùng ở bước này.
- `OUR-DE`: giống `OUR-holdout`, nhưng thay AOO bằng Differential Evolution (scipy).

### 3.1 Trung bình 5 POS (dữ liệu seed 0, giống thiết lập của bài)

| Mô hình | Bài: MSE | MAE | R² | Ta: MSE | MAE | R² |
|---|---|---|---|---|---|---|
| RF | 19.77 | 3.03 | 0.984 | **3.94** | 1.55 | 0.997 |
| RBF | 14.02 | 2.93 | 0.990 | 5.05 | 1.78 | 0.996 |
| KELM | 30.81 | 4.70 | 0.978 | 17.11 | 3.41 | 0.986 |
| ELM | 14.87 | 2.93 | 0.989 | 4.56 | 1.70 | 0.996 |
| ARIMA | 7.49 | 2.16 | 0.994 | 8.05 | 2.35 | 0.994 |
| **OUR** | **4.17** | 1.52 | 0.997 | **1.5·10⁷** | 896 | −12130 |
| OUR-holdout | - | - | - | 5.92 | 1.95 | 0.995 |
| OUR-DE | - | - | - | 6.31 | 2.02 | 0.995 |

### 3.2 Độ ổn định qua 5 bộ dữ liệu (seed dữ liệu 0-4), MSE trung bình 5 POS

| Seed | RF | RBF | KELM | ELM | ARIMA | OUR | OUR-holdout | OUR-DE |
|---|---|---|---|---|---|---|---|---|
| 0 | **3.94** | 5.05 | 17.11 | 4.56 | 8.05 | 1.5e7 | 5.92 | 6.31 |
| 1 | **4.52** | 5.45 | 22.93 | 5.90 | 7.66 | 5.7e7 | 5.69 | 6.22 |
| 2 | 4.98 | **4.96** | 22.39 | 5.31 | 10.49 | 1.1e7 | 7.48 | 5.09 |
| 3 | **4.87** | 7.39 | 23.61 | 7.52 | 9.97 | 5.2e8 | 8.55 | 15.66 |
| 4 | 5.35 | **5.27** | 19.83 | 6.07 | 6.72 | 3.2e11 | 5.87 | 9.39 |
| Trung vị | **4.87** | 5.27 | 22.39 | 5.90 | 8.05 | 5.7e7 | 5.92 | 6.31 |

Tính trên 25 cặp (seed, POS), số lần mỗi mô hình có MSE thấp nhất: RF 10, RBF 5, ELM 5, ARIMA 3, OUR-DE 2, OUR 0, OUR-holdout 0.

**Nhận xét**
- **Sàn nhiễu.** Nhiễu `2·N(0,1)` có phương sai 4, nên không mô hình nào có thể đạt MSE thấp hơn khoảng 4 một cách bền vững. MSE 4.17 của bài và 3.9-6 của các mô hình tốt bên ta đều đã chạm sàn này. Với chỉ 20 điểm kiểm định, chênh lệch giữa 4 và 6 phần lớn là do ngẫu nhiên.
- **Làm đúng như bài thì RBF-MKELM thất bại.** Dò 7 siêu tham số chỉ theo RMSE tập train sẽ đẩy mô hình về phía học thuộc dữ liệu train (C lớn, kernel rất hẹp): RMSE train gần 0, nhưng dự báo tập kiểm định vọt lên hàng triệu. Hiện tượng này lặp lại ở cả 5 seed dữ liệu, nên không phải do một seed xui. Bài báo được MSE 4.17 thì hoặc đã dùng thêm một cơ chế chống quá khớp không được mô tả, hoặc đã giới hạn miền tham số khác đi.
- **Sửa quá khớp xong thì RBF-MKELM chỉ ngang mô hình cơ bản.** Khi chọn tham số trên phần holdout, MSE còn khoảng 6. Mức này ngang RBF và ELM, kém RF, và không thắng ở seed nào. **Ta không tái lập được kết luận "RBF-MKELM tốt nhất" của bài.**
- **Thứ hạng các baseline lệch so với bài.** Bài cho RF tệ (19.8) còn ARIMA tốt thứ hai (7.5). Bên ta RF tốt nhất, ARIMA ở giữa, và ARIMA của ta khớp số của bài (8.05 so với 7.49). Bài có thể đã đặt tham số RF/RBF/ELM kém hơn mặc định của scikit-learn. Ở bảng của bài, chỉ riêng POS1 của RF đã có MSE 80.
- **AOO và DE cho kết quả gần nhau** (5.9 so với 6.3; mỗi bên thắng ở một vài seed). Ở bài toán này, thuật toán tối ưu "mới" không đem lại lợi thế rõ ràng. Lưu ý: bản AOO của ta dựng lại từ mô tả gián tiếp (bài RLDN-AOO, ERA 2025), một số hệ số là ước đoán.

## 4. Bảng 4 và 5 - So sánh 10 thuật toán phân bổ

*Đang chạy.* Xem trước ở quy mô 350-400 (một loại task, 10 lần chạy):

| n | Thuật toán | Bài: chi phí | Ta: chi phí | Bài: thời gian (s) | Ta: thời gian (s) |
|---|---|---|---|---|---|
| 400 | OUR | 1736.7 | 1673.1 | 7.5 | 18.3 |
| 400 | KCA | 2159.7 | 1685.7 | 4.1 | 14.6 |
| 400 | 3D-SOM | 1926.0 | 1722.6 | 0.7 | 2.9 |
| 400 | DQN | 3882.6 | 2118.2 | 605.4 | 38.0 |
| 400 | GA | 3788.1 | 3123.5 | 62.3 | 78.0 |

Nhận xét sơ bộ: OUR vẫn tốt nhất, nhưng KCA (K-means + đấu giá) chỉ kém khoảng 1%, trong khi bài báo cho khoảng cách 24%.
Bảng đầy đủ và kết luận sẽ bổ sung khi chạy xong.

## 5. Mô phỏng động (tương tự Hình 14)

Thiết lập (ta chọn, vì bài ghi pin "khoảng 10 km" trên bản đồ 100×100 km, hai số này mâu thuẫn nhau):
- Pin đủ cho 350 đơn vị quãng đường. UAV về sạc khi pin còn 20%, sạc mất 50 bước.
- Ở t = 101 thêm 20 task loại C.
- Khi RBF-MKELM dự báo nhu cầu ≥ 90 thì sinh task tiếp tế.
- Mỗi sự kiện kích hoạt phân bổ lại hai tầng.

| Mốc | Diễn biến |
|---|---|
| t = 1 | Phân bổ 100 task cho 5 UAV |
| t = 101 | Thêm 20 task loại C, phân bổ lại |
| t = 150-182 | 5 điểm cứu trợ đạt ngưỡng dự báo (95-103), sinh 5 task tiếp tế |
| t = 289-293 | Cả 5 UAV gần như cùng lúc về sạc (pin khoảng 70/350) |
| t = 374-392 | Các UAV sạc xong và quay lại |
| t = 455-482 | Đợt tiếp tế thứ hai |
| t = 544 | Hoàn thành 130 task, tổng quãng đường bay 2095.3 |

**Nhận xét**
- Cơ chế "sự kiện → phân bổ lại" hoạt động như bài mô tả.
- Có một điểm bài không nhắc tới: vì đấu giá cân bằng quãng đường rất tốt, **cả 5 UAV cạn pin gần như cùng lúc** (t = 289-293). Trong khoảng 80 bước sau đó không có UAV nào hoạt động, 25 task phải chờ. Cân bằng tải tốt về chi phí lại gây hại ở tính liên tục của dịch vụ. Đây là một điểm có thể khai thác trong hướng nghiên cứu của dự án: so le lịch sạc, hoặc thêm ràng buộc về số UAV tối thiểu luôn hoạt động.

## 6. Tóm tắt những gì khớp và không khớp (Bảng 1, 2, 7)

| Kết luận của bài | Tái lập | Ghi chú |
|---|---|---|
| α lớn làm UAV chuyên về một loại task | Khớp về chiều hướng, lệch về mức độ | Ở bản của ta α = 3 gây 2 UAV không nhận task nào; bài không gặp hiện tượng này |
| Đấu giá giảm chi phí trước/sau | Khớp | Mức cải thiện TB 5.4% so với 12.7% của bài |
| RBF-MKELM + AOO dự báo tốt nhất | **Không khớp** | Làm đúng quy trình của bài thì quá khớp; sửa xong chỉ ngang RF/RBF/ELM |
| Dự báo kích hoạt tiếp tế, hệ thống phản ứng với sự kiện | Khớp | Phát hiện thêm: các UAV cạn pin đồng loạt |

**Giới hạn của bản tái lập**
- Python thay vì MATLAB, nên thời gian chạy không so trực tiếp được.
- Những chỗ bài mơ hồ (chuẩn hóa α, Length, pin, chu kỳ F4, công thức AOO) là do ta tự chọn; mỗi lựa chọn đã ghi ở trên.
- Kịch bản do ta sinh ngẫu nhiên, không trùng với dữ liệu của bài, nên chỉ nên so xu hướng và % cải thiện.
