# Độ phức tạp và tính khó (T10)

* **Tính NP-khó.** Với một UAV, không có risk map, bỏ pin và cửa sổ thời gian, bài toán rút gọn về TSP; với K UAV có tải và cửa sổ thời gian thì là VRPTW, NP-khó theo nghĩa mạnh (Solomon 1987; Toth & Vigo 2014). Mô hình ở `mo_hinh_toan.md` chứa VRPTW làm trường hợp đặc biệt (không có hotspot, pin vô hạn, mọi điểm biết từ đầu), nên cũng NP-khó. Giải chính xác chỉ khả thi ở cỡ nhỏ (vài chục điểm), vì vậy dùng heuristic ALNS.
* **Thêm tính động** (điểm mới, đổi deadline, hạ pin, đổi risk map): mỗi lần kích hoạt phải giải lại một bài toán cùng độ khó trong thời gian ngắn → cần heuristic nhanh và khởi động ấm.
* **Chi phí mỗi thành phần** (n khách, K UAV, L = độ dài tuyến, S = 6 mẫu/chặng, B = số thùng thời gian):
  * Dựng bảng rủi ro/cấm `X, BAN, CR` cho một snapshot: `O((n+K)^2 · S)` truy vấn trường rủi ro, làm một lần mỗi lần lập lại kế hoạch.
  * `eval_route` (chính xác): `O(L)`.
  * Quét chèn nhanh một khách vào một tuyến (`scan_route`, push-forward): `O(L)`; chèn vào mọi tuyến: `O(n)`; một lần repair với q khách bị gỡ: `O(q · n)` cộng `q` lần kiểm tra chính xác. Regret-2 thêm hệ số `q` do phải tính cho mọi khách còn lại mỗi bước: `O(q² · n)`.
  * Destroy: random/worst/risk `O(n)`; related (Shaw) `O(n log n)`.
  * Một vòng ALNS ≈ `O(q · n)`, tổng `O(iters · q · n)`; thực nghiệm (`results/tuning_alns.csv`) thời gian tăng gần tuyến tính theo số vòng và nhanh hơn tuyến tính theo n (1000 vòng, lần chạy 17/11/2026 trên dữ liệu thật: 2,2 s tại n=25, 6,2 s tại n=50, 31–35 s tại n=100, 46–51 s tại n=200; số tuyệt đối phụ thuộc máy, lần chạy cũ nhanh hơn khoảng 6–7 lần).
* **Giá trị thực tiễn.** Gap tới nghiệm 1000 vòng cho thấy ngân sách 100 vòng chưa đủ cho n ≥ 100 (gap 27–34%), nên chọn 150/250/400/600 vòng mỗi lần giải theo cỡ instance; 400 vòng ở n = 100 là ngân sách yếu nhất (`docs/tuning_alns.md`).
