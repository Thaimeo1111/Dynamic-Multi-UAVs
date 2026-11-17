# Ngân sách tính toán (T25)

Môi trường: 2 nhân CPU, 7 GB RAM, không GPU.

| Thí nghiệm | Số job | Cấu hình |
|---|---|---|
| main | 960 | 12 instance Solomon gốc (n = 25, 50, 100) × 4 thuật toán × 20 seed, cường độ med |
| ablation | 400 | 4 instance (n ≤ 50) × 5 biến thể × 20 seed |
| large | 60 | 3 instance Gehring–Homberger 200 khách (loại 1) × 4 thuật toán × 5 seed, pin 450 |
| sensitivity_large | 135 | 3 instance Homberger 200 khách × 5 seed × 9 mức (theta_ban 4, w_r 5), pin 450 |
| sensitivity | 576 | 4 instance (n ≤ 50) × 6 seed × 24 mức tham số (dt, cường độ, w_r, theta_ban, số vòng, q_hi) |

**Thời gian đo được** (cộng thời gian của từng job; chia cho 2 worker ra thời gian thực tế tối thiểu):
main ≈ 0,76, ablation ≈ 0,12, sensitivity ≈ 0,18, large ≈ 0,67, sensitivity_large ≈ 2,37 giờ-lõi (tổng khoảng 4,1). Homberger 200 khách nặng hơn nhiều (đo từ cột `wall` của `results/raw_large.csv`): trung bình mỗi job `greedy` 2,4 s, `alns_static` 26 s, `resolve` 64 s, `alns_dynamic` 68 s; job chậm nhất khoảng 105 s (`resolve`) và 111 s (`alns_dynamic`). Vì vậy chỉ chạy 5 seed.

**Đã thu nhỏ so với kế hoạch đầy đủ:** số vòng ALNS mỗi lần giải 150/250/400/600 (không phải hàng nghìn); ablation và sensitivity chỉ trên n ≤ 50; Homberger chỉ 3 instance × 5 seed (độ nhạy ở 200 khách chỉ theta_ban và w_r); không dùng instance loại 2.

**Chạy cấu hình lớn hơn:** sửa `config.yaml` (danh sách instance, `alns.iters`, `seeds`) rồi chạy `./run_all_experiments.sh`; có `--resume` nên ngắt và chạy tiếp được. Thời gian tăng gần tuyến tính theo (số job × số vòng ALNS).
