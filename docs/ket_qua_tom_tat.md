# Tóm tắt kết quả (dữ liệu thật: Solomon gốc, Gehring–Homberger, nhãn RescueNet; số liệu từ results/*.csv, tables/*.md)

**Chính (960 run: 12 instance × 4 thuật toán × 20 seed).** J_ref trung bình: greedy 3981, alns_static 2947, resolve 2472, alns_dynamic 2201 (−45%, −25%, −11%). Friedman p≈5e-102; Wilcoxon+Holm p<1e-22; effect size so với resolve nhỏ (A12=0,55, Cliff δ=−0,11).

**Homberger 200 khách (60 run, 5 seed, pin 450).** greedy 10715, static 6913, resolve 6014, dynamic 5840; dynamic hơn resolve 2,9%, thắng 8/15 cặp, p=0,21 (chưa có ý nghĩa).

**Ablation (400 run).** Bỏ rolling +20,3%; giải lại nguội cùng chu kỳ +12,8%; bỏ warm-start +10,7%; bỏ risk-priority repair +0,2%; bỏ risk-based removal −0,4% (hai toán tử rủi ro không có đóng góp đo được).

**Độ nhạy (576 run).** theta_ban nhạy nhất (J 1378→2279 khi 0,9→0,6); w_r chữ U (tốt ở 5–10; 20 bỏ rơi khách); cường độ sự kiện low→high 1322→1563; q_hi, số vòng, dt bền (lệch <5%).

**Độ nhạy 200 khách (135 run).** theta_ban 0,6→0,9: J 6529→5555 (cùng hướng với n ≤ 50). w_r 0→20: J 5733→6555, tăng đều (không có dạng chữ U); rủi ro giảm 251→170 nhưng trễ tăng 368→1054, là đánh đổi.

**NSGA-III vs ALNS archive (tĩnh, 3 instance nhỏ × 10 seed).** HV 2,40e8 vs 0,81e8; IGD 947 vs 1426; ALNS hơn ở 20/30 cặp (p=0,003). NSGA-III là bản tự viết, chưa tinh chỉnh.

**RescueNet (4 mặt nạ × 3 seed, RC101_050).** greedy 8421, static 6548, resolve 5467, dynamic 5566; hai bản động ngang nhau. Mức rủi ro theo lớp là giả định; mặt nạ 10801 bị loại (suy biến).

Kết quả của đợt chạy đầu trên dữ liệu R/RC tự sinh nằm ở `results/synthetic_v1/` (kết luận định tính giống nhau).
