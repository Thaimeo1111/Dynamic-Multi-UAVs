# Gói nộp Tuần 1–2 (T11)

**Nội dung:** (1) mô hình toán → `docs/mo_hinh_toan.md`; (2) benchmark nhỏ → `data/solomon_small/` (Solomon gốc C101, C109, C106, R101, RC101 cắt 25/50 khách đầu); (3) bộ sinh risk map, sự kiện, evaluator, 3 baseline → `src/`; (4) kết quả baseline → bảng dưới + `results/baseline_small.csv`; (5) độ phức tạp → `docs/do_phuc_tap.md`.

**Bài toán.** Đội UAV đồng nhất giao hàng/cứu trợ tới các điểm có cửa sổ thời gian; điểm mới xuất hiện dần, deadline và pin đổi, vùng nguy hiểm (hotspot) di chuyển. Mục tiêu J = quãng đường + 5·rủi ro phơi nhiễm + trễ + 300·điểm bỏ sót, ràng buộc cứng: pin (gồm đường về), tải, vùng cấm, về trước hết ngày.

**Ba baseline** (chạy đồng thời với bản động ở tuần 3–6): `greedy` (chèn rẻ nhất, không tối ưu lại), `alns_static` (ALNS một lần lúc t=0 trên khách đã biết, khách đến sau chỉ chèn tham lam), `resolve` (ALNS giải lại từ đầu ở mỗi sự kiện).

## Kết quả trên 6 instance nhỏ × 5 seed (J càng thấp càng tốt)

| Instance | Thuat toan | J (TB +- std) | Quang duong | Rui ro | Tre | Khong PV | Runtime ke hoach (s) | Wall (s) |
|---|---|---|---|---|---|---|---|---|
| C101_025 | greedy | 1295 +- 445 | 367 | 61 | 382 | 0.8 | 0.11 | 0.12 |
| C101_025 | alns_static | 877 +- 228 | 270 | 58 | 139 | 0.6 | 0.21 | 0.22 |
| C101_025 | resolve | 854 +- 193 | 284 | 62 | 140 | 0.4 | 0.51 | 0.52 |
| R101_025 | greedy | 3222 +- 250 | 398 | 47 | 369 | 7.4 | 0.01 | 0.02 |
| R101_025 | alns_static | 2875 +- 307 | 382 | 36 | 335 | 6.6 | 0.10 | 0.11 |
| R101_025 | resolve | 2449 +- 316 | 389 | 39 | 304 | 5.2 | 0.27 | 0.28 |
| RC101_025 | greedy | 1587 +- 819 | 373 | 51 | 59 | 3.0 | 0.01 | 0.02 |
| RC101_025 | alns_static | 1514 +- 813 | 395 | 57 | 56 | 2.6 | 0.10 | 0.11 |
| RC101_025 | resolve | 1576 +- 1170 | 414 | 69 | 39 | 2.6 | 0.22 | 0.23 |
| C101_050 | greedy | 2557 +- 660 | 773 | 127 | 970 | 0.6 | 0.24 | 0.25 |
| C101_050 | alns_static | 1542 +- 500 | 566 | 103 | 398 | 0.2 | 0.88 | 0.89 |
| C101_050 | resolve | 1496 +- 477 | 643 | 121 | 250 | 0.0 | 2.01 | 2.03 |
| R101_050 | greedy | 5031 +- 903 | 796 | 84 | 754 | 10.2 | 0.03 | 0.05 |
| R101_050 | alns_static | 3793 +- 688 | 790 | 97 | 718 | 6.0 | 0.56 | 0.58 |
| R101_050 | resolve | 3318 +- 408 | 776 | 80 | 822 | 4.4 | 1.10 | 1.12 |
| RC101_050 | greedy | 2678 +- 1264 | 878 | 106 | 371 | 3.0 | 0.03 | 0.05 |
| RC101_050 | alns_static | 2115 +- 577 | 890 | 110 | 314 | 1.2 | 0.53 | 0.55 |
| RC101_050 | resolve | 1898 +- 455 | 899 | 112 | 257 | 0.6 | 1.15 | 1.17 |

**Đọc kết quả:** `resolve` có J thấp nhất ở 5/6 instance (ngoại lệ RC101_025: static 1514 < resolve 1576, chênh nhỏ so với độ lệch chuẩn), `greedy` tệ nhất ở cả 6; chênh lệch lớn nhất ở R101 (cửa sổ thời gian chặt, nhiều điểm không phục vụ vì horizon chỉ 230). Runtime tăng theo thứ tự greedy < static < resolve (vài giây ở n=50).

**Giới hạn:** chỉ 5 seed, std lớn; `R101_025` bỏ sót 5–7/25 điểm vì cửa sổ ngắn so với pin/horizon, không phải lỗi thuật toán.
