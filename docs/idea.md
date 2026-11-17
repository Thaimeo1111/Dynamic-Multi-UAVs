# Ý tưởng nghiên cứu (T20)

**Câu hỏi.** Khi mục tiêu xuất hiện dần, deadline và pin đổi, và vùng nguy hiểm (risk map) di chuyển ngoài dự đoán, một bộ điều phối đa UAV lập lại kế hoạch kiểu *rolling-horizon* với ALNS nhận biết rủi ro có tốt hơn (i) kế hoạch tĩnh + chèn tham lam và (ii) giải lại từ đầu mỗi sự kiện hay không?

**Đóng góp dự kiến**
1. Mô hình dVRPTW cho đa UAV với phơi nhiễm rủi ro theo thời gian, phân biệt *truth* và *belief* (`docs/mo_hinh_toan.md`).
2. Bộ điều phối ALNS động: khởi động ấm từ kế hoạch đang chạy, toán tử destroy/repair có nhận biết rủi ro (risk-based removal, risk-priority repair), kích hoạt theo sự kiện + chu kỳ.
3. Quy trình đánh giá tái lập được: hạt giống cố định, 20 seed, ablation, độ nhạy, kiểm định thống kê.
4. Quan điểm đa mục tiêu (quãng đường / rủi ro / trễ) so với NSGA-III ở bài toán tĩnh bằng HV và IGD.
5. (Hướng mở) Lớp rủi ro tĩnh từ mặt nạ phân đoạn thảm họa kiểu RescueNet (`src/rescuenet_risk.py`), hiện chỉ minh họa bằng mặt nạ giả vì chưa có dữ liệu thật.

**So sánh với Pu et al. (2026):** cùng hướng bài toán, nhưng mô hình và instance của ta là tái hiện đơn giản, không phải tái lập số liệu của họ; chỉ so sánh định tính.

**Giả thuyết kiểm định.** H1: `alns_dynamic` có J thấp hơn `resolve`, `alns_static`, `greedy` có ý nghĩa thống kê. H2: bỏ warm-start hoặc rolling làm J tăng. H3: lợi ích tăng theo cường độ sự kiện. H4: hiệu quả tăng theo w_r đến một ngưỡng rồi bão hòa (đánh đổi quãng đường).
