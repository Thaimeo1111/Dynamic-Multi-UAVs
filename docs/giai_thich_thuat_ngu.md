# Giải thích thuật ngữ (hỏi đáp)

Ghi lại các câu hỏi về tài liệu tổng hợp dự án
(claude.ai/artifact/H7ZUS8M18FAHW2QZpaPeb5) và lời giải thích.
Mỗi mục có dẫn tới file nguồn để kiểm tra lại.

Mục lục
- 1. Rolling-horizon là gì?
- 2. J là gì? (và J_ref)
- 3. Vì sao 960 lần chạy, vì sao 20 seed?
- 4. ALNS là gì?
- 5. Warm-start là gì?
- 6. Bốn thuật toán so sánh khác nhau thế nào?
- 7. Greedy chèn "chỗ rẻ nhất" là thế nào?
- 8. Belief và truth khác nhau thế nào?
- 9. Hotspot là gì? theta_ban là gì?
- 10. Vòng lặp phá - sửa trong ALNS; vì sao greedy = "sửa một lần"?
- 11. Regret-2 là gì?
- 12. Các thuật toán trong bài Pu et al. (2026) - docs/1.pdf
- 13. VRPTW / dVRPTW là gì? Instance Solomon là gì?
- 14. Ablation và độ nhạy là gì?
- 15. Kiểm định thống kê: Friedman, Wilcoxon, Holm, p_holm, A12
- 16. Bảng giả thuyết H1-H4 đọc thế nào?
- 17. Đa mục tiêu, NSGA-III, HV và IGD
- 18. RescueNet và "mặt nạ phân đoạn"
- 19. "Tái hiện đơn giản, chỉ so sánh định tính" nghĩa là gì?

---

## 1. Rolling-horizon là gì?

Hỏi ngày 2026-11-17.

Hiểu đơn giản: LẬP KẾ HOẠCH -> BAY MỘT ĐOẠN -> LẬP LẠI KẾ HOẠCH, lặp đến hết ngày.

Trong bài toán của bạn, thông tin không có đủ từ đầu: điểm cần giao mới xuất hiện,
deadline thay đổi, pin tụt, vùng nguy hiểm di chuyển. Vì vậy không thể lập một kế hoạch
duy nhất lúc t = 0 rồi bay theo đến cuối ngày.

Các bước (src/rolling_horizon.py):

  1. Tại thời điểm τ, chụp lại trạng thái hiện tại: UAV ở đâu, còn bao nhiêu pin/tải,
     còn điểm nào chưa phục vụ, bản đồ rủi ro lúc đó trông thế nào.
  2. Giải bài toán định tuyến (bằng ALNS) CHỈ cho phần còn lại.
  3. Cho UAV bay theo kế hoạch đó.
  4. Lập lại kế hoạch khi có sự kiện mới (hai lần cách nhau tối thiểu 40 đơn vị
     thời gian) hoặc định kỳ mỗi dt_replan = 100 (config.yaml:17).
  5. Phần đã bay hoặc đang phục vụ được "ĐÓNG BĂNG": không bao giờ bị sửa lại
     (docs/mo_hinh_toan.md:44).

Ví dụ: giống dùng Google Maps khi lái xe. Bạn có tuyến ban đầu, gặp tắc đường thì bản đồ
tính lại tuyến từ vị trí hiện tại. Đoạn đường đã đi thì không tính lại.

Tên gọi: "horizon" (tầm nhìn) "rolling" (trượt) nghĩa là cửa sổ lập kế hoạch luôn bắt đầu
từ thời điểm hiện tại và trượt dần theo thời gian.

Vai trò trong dự án: đây là cốt lõi của thuật toán alns_dynamic. Thí nghiệm ablation cho
thấy bỏ rolling thì J tăng 20,3% (docs/ket_qua_tom_tat.md:7).

---

## 2. J là gì?

Hỏi ngày 2026-11-17.

J là TỔNG CHI PHÍ của một phương án bay, CÀNG THẤP CÀNG TỐT (docs/mo_hinh_toan.md:65):

  J = 1 x quãng đường
    + 5 x rủi ro phơi nhiễm
    + 1 x tổng thời gian trễ
    + 300 x số điểm không phục vụ được

  | Thành phần      | Ý nghĩa                                              | Trọng số |
  |-----------------|------------------------------------------------------|----------|
  | Quãng đường     | tổng quãng đường các UAV bay                         | w_d = 1  |
  | Rủi ro          | bay qua vùng nguy hiểm càng lâu, càng gần tâm thì    | w_r = 5  |
  |                 | càng cao (rủi ro bị coi nặng hơn quãng đường)        |          |
  | Trễ             | tổng thời gian đến muộn so với deadline              | w_l = 1  |
  | Bỏ sót          | mỗi điểm không phục vụ được; phạt rất nặng để buộc   | w_u = 300|
  |                 | thuật toán cố phục vụ hết                            |          |

Gộp 4 tiêu chí thành một con số giúp so sánh thuật toán dễ dàng.
Ví dụ: greedy J = 3981, alns_dynamic J = 2201 nghĩa là alns_dynamic tốt hơn khoảng 45%.

### J_ref (trong bảng kết quả)

- J được tính sau khi UAV BAY THẬT trên bản đồ rủi ro thật (truth), không phải trên bản
  đồ dự đoán (belief) của thuật toán.
- J_ref luôn dùng bộ trọng số mặc định (1, 5, 1, 300) (src/experiments.py:116).
  Nhờ vậy, ngay cả trong thí nghiệm độ nhạy (khi đổi w_r), các kết quả vẫn so sánh được
  trên cùng một thước đo.

---

## 3. Vì sao 960 lần chạy, vì sao 20 seed?

Hỏi ngày 2026-11-17.

  960 = 12 instance x 4 thuật toán x 20 seed     (docs/ngan_sach_tinh_toan.md:7)

- 12 instance  : 12 bộ dữ liệu Solomon (25, 50, 100 khách; nhóm C, R, RC).
- 4 thuật toán : greedy, alns_static, resolve, alns_dynamic.
- 20 seed      : 20 kịch bản ngẫu nhiên khác nhau cho mỗi instance.

### Seed là gì?

Seed (hạt giống ngẫu nhiên) là con số khởi tạo bộ sinh số ngẫu nhiên. Cùng một seed thì
luôn ra cùng một kết quả ngẫu nhiên, nên thí nghiệm TÁI LẬP ĐƯỢC.

Trong dự án, seed quyết định hai thứ:

  1. Kịch bản sự kiện (src/event_generator.py:34-35): điểm nào xuất hiện muộn và lúc
     nào, vùng nguy hiểm sinh ra ở đâu và rẽ hướng khi nào, deadline nào đổi, pin tụt
     lúc nào. Seed 1 là một "ngày bay" khác hẳn seed 2.
  2. Tính ngẫu nhiên bên trong ALNS (ALNS chọn toán tử một cách ngẫu nhiên).

Với cùng instance và cùng seed, cả 4 thuật toán gặp ĐÚNG CÙNG một kịch bản
(src/event_generator.py:8), nên so sánh là công bằng.

### Vì sao cần 20 seed mà không chạy 1 lần?

- Kết quả phụ thuộc vào may rủi của kịch bản: thuật toán có thể thắng ở seed này nhưng
  thua ở seed khác. Chạy 20 kịch bản rồi lấy trung bình mới cho kết luận đáng tin.
- Đủ mẫu mới làm được kiểm định thống kê (Friedman, Wilcoxon), tức là trả lời câu hỏi
  "thắng thật hay chỉ do may".
- Ví dụ: ở tuần 1-2 chỉ có 5 seed nên độ lệch chuẩn lớn, kết luận còn yếu
  (docs/goi_nop_tuan_1_2.md:34).

Ngoại lệ: Homberger 200 khách chỉ chạy 5 seed vì mỗi lần chạy mất khoảng 68-110 giây,
chạy 20 seed quá tốn thời gian (docs/ngan_sach_tinh_toan.md:14).

---

## 4. ALNS là gì?

Hỏi ngày 2026-11-17.

ALNS = Adaptive Large Neighborhood Search (Tìm kiếm lân cận lớn thích nghi),
do Ropke & Pisinger đề xuất năm 2006. Đây là thuật toán dùng để tìm tuyến bay tốt cho
các UAV (code chính: src/alns_dynamic.py).

Ý tưởng cốt lõi: PHÁ MỘT PHẦN -> SỬA LẠI -> GIỮ NẾU TỐT HƠN, lặp nhiều vòng.

Một vòng lặp (src/alns_dynamic.py:89-135):

  1. Lấy phương án hiện tại (các tuyến của các UAV).
  2. PHÁ (destroy): gỡ khoảng 10-30% số điểm ra khỏi tuyến (tối đa 30 điểm).
  3. SỬA (repair): chèn lại các điểm vừa gỡ vào vị trí mới.
  4. Tính J của phương án mới, rồi quyết định có nhận nó hay không.

Ví dụ: xếp lịch giao hàng cho 3 xe. Bạn rút 5 đơn ra khỏi lịch rồi xếp lại 5 đơn đó
theo cách khác. Nếu lịch mới tốt hơn thì giữ. Làm vài trăm lần thì lịch tốt dần lên.

### Các toán tử phá và sửa

  | Toán tử PHÁ (src/destroy_ops.py) | Gỡ điểm nào?                                  |
  |----------------------------------|-----------------------------------------------|
  | random                           | ngẫu nhiên                                    |
  | worst                            | điểm đang "tốn" nhất (bỏ đi thì J giảm nhiều) |
  | related                          | các điểm gần nhau (để xếp lại cả cụm)         |
  | risk                             | điểm làm UAV bay qua vùng rủi ro cao          |

  | Toán tử SỬA (src/repair_ops.py)  | Chèn lại thế nào?                             |
  |----------------------------------|-----------------------------------------------|
  | greedy                           | chèn vào chỗ làm J tăng ít nhất               |
  | regret2                          | ưu tiên điểm "nếu không chèn ngay thì sau sẽ  |
  |                                  | rất thiệt" (khó xếp)                          |
  | riskprio                         | ưu tiên điểm nằm trong vùng rủi ro            |

Hai toán tử "risk" và "riskprio" là phần nhận biết rủi ro do dự án thêm vào.

### Chữ "Adaptive" (thích nghi) nghĩa là gì?

Mỗi toán tử có một trọng số. Toán tử nào hay tạo ra phương án tốt thì được cộng điểm
(33 điểm nếu tìm ra kỷ lục mới, 9 nếu tốt hơn hiện tại, 3 nếu được nhận dù kém hơn).
Cứ 50 vòng, trọng số được cập nhật theo điểm, rồi toán tử được chọn bằng "vòng quay"
theo trọng số: toán tử hiệu quả được dùng nhiều hơn (src/alns_dynamic.py:119-130).

### Vì sao đôi khi nhận phương án kém hơn?

Dùng luật simulated annealing (mô phỏng tôi luyện): lúc đầu "nhiệt độ" cao nên đôi khi
nhận phương án tệ hơn một chút, để thoát khỏi chỗ "tốt cục bộ". Nhiệt độ giảm dần, về
cuối gần như chỉ nhận phương án tốt hơn (src/alns_dynamic.py:110-112).

### Chạy bao nhiêu vòng?

Mỗi lần giải: 150 / 250 / 400 / 600 vòng cho 25 / 50 / 100 / 200 khách (config.yaml:14).

### Các thuật toán trong dự án đều dùng ALNS, khác nhau ở cách dùng

  | Thuật toán   | Cách dùng ALNS                                                    |
  |--------------|-------------------------------------------------------------------|
  | greedy       | không dùng ALNS, chỉ chèn tham lam (mốc so sánh yếu nhất)         |
  | alns_static  | giải ALNS MỘT lần lúc t = 0 (gấp 4 số vòng), sau đó chèn tham lam |
  | resolve      | giải lại ALNS cơ bản TỪ ĐẦU mỗi khi có sự kiện                    |
  | alns_dynamic | rolling-horizon + ALNS đầy đủ (toán tử rủi ro, thích nghi,        |
  |              | warm-start)                                                       |

---

## 5. Warm-start là gì?

Hỏi ngày 2026-11-17.

Warm-start (khởi động ấm) = mỗi lần lập lại kế hoạch, ALNS BẮT ĐẦU TỪ KẾ HOẠCH CŨ thay vì
xây lại từ con số 0.

  - Cold start (nguội, không warm-start): xóa hết tuyến cũ, xây phương án ban đầu từ
    đầu bằng regret-2, rồi mới chạy ALNS.
  - Warm start (ấm): lấy các tuyến UAV đang bay (bỏ phần đã bay xong), chỉ chèn thêm
    các điểm mới/chưa xếp, rồi chạy ALNS từ đó.

Code: src/rolling_horizon.py:220 (hàm warm_routes lấy tuyến hiện tại),
      src/alns_dynamic.py:57-63 (initial_solution: warm thì dùng tuyến cũ, không thì
      bắt đầu từ phương án rỗng).

Ví dụ: đang có lịch giao hàng tốt cho 3 xe thì có thêm 2 đơn mới. Warm-start là giữ
nguyên lịch cũ và chèn 2 đơn vào. Cold-start là xé lịch, xếp lại toàn bộ.

### Vì sao warm-start tốt hơn?

1. Kế hoạch cũ đã được tối ưu qua nhiều vòng ALNS trước đó. Bắt đầu từ đó thì số vòng
   ALNS có hạn (vài trăm vòng) được dùng để cải thiện tiếp, thay vì tốn công tìm lại.
2. Kế hoạch ổn định hơn: ít đổi UAV phục vụ một điểm (chỉ số reassign thấp hơn).

Kết quả ablation (docs/ket_qua_tom_tat.md:7):
  - Bỏ warm-start (dyn_no_warm): J tăng 10,7%.
  - Giải lại nguội cùng chu kỳ (resolve_periodic): J tăng 12,8%.

Đây chính là điểm khác lớn giữa alns_dynamic và resolve: resolve giải lại TỪ ĐẦU
(nguội) mỗi khi có sự kiện, còn alns_dynamic tiếp tục từ kế hoạch cũ (ấm).

---

## 6. Bốn thuật toán so sánh khác nhau thế nào?

Hỏi ngày 2026-11-17 (về bảng 4 thuật toán ở mục 4).

Hình dung cả 4 như 4 người điều phối cùng trải qua MỘT ngày bay giống hệt nhau (cùng
seed). Khác nhau ở 3 câu hỏi:
  (a) Lập kế hoạch ban đầu tốt đến đâu?
  (b) Khi nào thì lập lại kế hoạch?
  (c) Lập lại bằng cách nào?

### greedy - "chỉ chèn thêm, không bao giờ sắp xếp lại"   (src/baseline_greedy.py)

- Lúc t = 0: chèn từng điểm vào chỗ rẻ nhất (cheapest insertion). Không tối ưu.
- Khi có ĐIỂM MỚI xuất hiện: chèn điểm đó vào tuyến đang có, chỗ làm chi phí tăng ít
  nhất. Không đụng tới các điểm đã xếp.
- Khi vùng nguy hiểm đổi, deadline đổi, pin tụt: KHÔNG phản ứng (chỉ có phản xạ an toàn
  chung: thiếu pin thì bỏ điểm).
- Vai trò: mốc "không tối ưu", yếu nhất. Nếu thuật toán mới không hơn được greedy thì
  vô nghĩa.

### alns_static - "lập kế hoạch thật kỹ một lần, sau đó chỉ vá"   (src/alns_static.py)

- Lúc t = 0: giải bằng ALNS cơ bản với số vòng GẤP 4 lần bình thường (config
  static_mult = 4), nên kế hoạch ban đầu rất tốt.
- Sau đó: giống greedy, điểm mới chỉ được chèn tham lam; không phản ứng với thay đổi
  rủi ro/deadline/pin.
- Khác greedy duy nhất ở chỗ: kế hoạch ban đầu do ALNS tối ưu.
- Vai trò: trả lời câu hỏi "có cần lập lại kế hoạch không, hay chỉ cần kế hoạch ban đầu
  tốt là đủ?".

### resolve - "có biến là xé kế hoạch, làm lại từ đầu"   (src/baseline_resolve.py)

- Lập lại kế hoạch MỖI KHI CÓ SỰ KIỆN: điểm mới, vùng nguy hiểm xuất hiện/rẽ hướng,
  đổi deadline, pin tụt (src/rolling_horizon.py:230, chế độ "events").
- Mỗi lần lập lại: chạy ALNS CƠ BẢN từ con số 0 (cold start, không dùng kế hoạch cũ):
  không có toán tử rủi ro, không thích nghi.
- Vai trò: đối thủ mạnh nhất. Đây là cách làm "hiển nhiên" khi có thay đổi.

### alns_dynamic - phương pháp đề xuất của dự án   (src/alns_dynamic.py)

- Lập lại kế hoạch mỗi khi có sự kiện (như resolve) VÀ thêm định kỳ mỗi 100 đơn vị
  thời gian (rolling-horizon, dt_replan = 100).
- Mỗi lần lập lại: chạy ALNS ĐẦY ĐỦ:
    * warm-start: tiếp tục từ kế hoạch cũ thay vì làm lại;
    * thích nghi: toán tử nào hiệu quả được dùng nhiều hơn;
    * toán tử rủi ro: "risk" (gỡ điểm ở vùng nguy hiểm) và "riskprio" (ưu tiên chèn
      điểm ở vùng nguy hiểm).

### Bảng tóm tắt

  | Thuật toán   | Kế hoạch ban đầu | Lập lại khi nào?          | Lập lại bằng cách nào?    |
  |--------------|------------------|---------------------------|---------------------------|
  | greedy       | chèn tham lam    | chỉ khi có điểm mới       | chèn tham lam điểm mới    |
  | alns_static  | ALNS, 4x vòng    | chỉ khi có điểm mới       | chèn tham lam điểm mới    |
  | resolve      | ALNS cơ bản      | mọi sự kiện               | ALNS cơ bản, từ đầu       |
  | alns_dynamic | ALNS đầy đủ      | mọi sự kiện + mỗi 100     | ALNS đầy đủ, từ kế hoạch  |
  |              |                  |                           | cũ (warm-start)           |

Mỗi hàng thêm một "nâng cấp" so với hàng trên, nên so sánh từng cặp cho biết nâng cấp đó
đáng giá bao nhiêu.

### Kết quả (J trung bình, 960 lần chạy; docs/ket_qua_tom_tat.md:3)

  greedy 3981  >  alns_static 2947  >  resolve 2472  >  alns_dynamic 2201

  alns_dynamic thấp hơn greedy 45%, alns_static 25%, resolve 11%.
  Lưu ý: so với resolve, khác biệt có ý nghĩa thống kê nhưng effect size NHỎ
  (A12 = 0,55): alns_dynamic chỉ thắng resolve khoảng 55% số cặp so sánh.

---

## 7. Greedy chèn "chỗ rẻ nhất" là thế nào?

Hỏi ngày 2026-11-17 (về greedy ở mục 6).
Code: src/repair_ops.py (scan_route dòng 18, _greedy_like dòng 136), src/baseline_greedy.py.

### 7.1 Lúc t = 0 có gì?

- Khoảng 70% điểm đã biết từ đầu; 30% còn lại sẽ xuất hiện dần trong ngày
  (online_frac = 0.3, config.yaml:12).
- Tất cả UAV đang đứng ở depot, tuyến rỗng: depot -> depot.
- Bản đồ rủi ro: dùng "belief" lúc t = 0, tức các vùng nguy hiểm đã biết, vị trí tương
  lai được đoán bằng cách kéo dài theo vận tốc hiện tại (docs/mo_hinh_toan.md, mục 1.2).

### 7.2 "Chỗ" là gì?

Một "chỗ" = một cặp (UAV nào, chèn vào giữa hai điểm nào trong tuyến của UAV đó).
Tuyến có L điểm thì có L + 1 chỗ để chèn.

  Ví dụ tuyến UAV 1:  depot -> A -> B -> depot
  Các chỗ để chèn điểm C:
    chỗ 0: depot -> C -> A -> B -> depot
    chỗ 1: depot -> A -> C -> B -> depot
    chỗ 2: depot -> A -> B -> C -> depot

### 7.3 Đánh giá một chỗ như thế nào? ("rẻ" = J tăng ít nhất)

Với mỗi chỗ, tính chi phí TĂNG THÊM khi chèn C vào giữa P (điểm trước) và N (điểm sau):

  Δ = 1 x (quãng đường tăng thêm)
    + 5 x (rủi ro tăng thêm)
    + 1 x (độ trễ tăng thêm)

  - Quãng đường tăng thêm = d(P,C) + d(C,N) - d(P,N)
    (bỏ chặng P->N, thay bằng hai chặng P->C và C->N)
  - Rủi ro tăng thêm = rủi ro chặng P->C + C->N - rủi ro chặng P->N.
    Rủi ro tính theo THỜI ĐIỂM UAV sẽ bay qua chặng đó (vì vùng nguy hiểm di chuyển).
  - Độ trễ tăng thêm = C có bị trễ deadline không, CỘNG độ trễ mà các điểm phía sau
    bị đẩy lùi (chèn C làm UAV đến N, rồi các điểm sau N, muộn hơn).

Ví dụ số (giả định):
  depot(0,0), A(10,0), B(20,0), C(10,5); tạm bỏ qua rủi ro và trễ.
  chỗ 0: d(depot,C)+d(C,A)-d(depot,A) = 11,2 + 5 - 10   = 6,2
  chỗ 1: d(A,C)+d(C,B)-d(A,B)         = 5 + 11,2 - 10   = 6,2
  chỗ 2: d(B,C)+d(C,depot)-d(B,depot) = 11,2 + 11,2 - 20 = 2,4  <- rẻ nhất về quãng đường
  Nếu chặng B->C đi qua vùng nguy hiểm (rủi ro tăng 2, nhân 5 = 10) thì chỗ 2 thành
  12,4 và chỗ 0 hoặc 1 lại rẻ hơn. Đó là cách rủi ro ảnh hưởng tới lựa chọn.

### 7.4 Chỗ bị loại (không khả thi)

Một chỗ bị bỏ qua, không xét, nếu:
  - pin không đủ cho cả tuyến (gồm cả đường về depot);
  - UAV hết tải;
  - một chặng mới đi qua VÙNG CẤM (rủi ro > theta_ban);
  - UAV về depot sau giờ kết thúc ngày bay.

### 7.5 Thứ tự chèn

Lặp lại cho tới khi hết điểm:
  1. Với MỖI điểm chưa xếp, tìm chỗ rẻ nhất của nó trên TẤT CẢ UAV.
  2. Trong tất cả các điểm, chọn điểm có Δ nhỏ nhất, chèn nó vào.
  3. Tính lại (vì tuyến vừa thay đổi), quay lại bước 1.
Nếu chỗ rẻ nhất của một điểm vẫn tốn >= 300 (w_u) thì BỎ điểm đó: bỏ chịu phạt 300
còn rẻ hơn chèn. Nếu không có chỗ khả thi nào thì cũng bỏ.

### 7.6 Vì sao "không tối ưu"?

- Thiển cận: mỗi bước chỉ nhìn bước đó. Điểm chèn trước chiếm chỗ tốt, điểm chèn sau
  phải chịu chỗ xấu.
- Đã chèn là CỐ ĐỊNH: không bao giờ gỡ ra xếp lại. ALNS thì làm đúng việc này
  (phá rồi sửa) hàng trăm lần, nên tìm được phương án tốt hơn nhiều.
- Thực ra bước "sửa greedy" trong ALNS chính là thủ tục này. Greedy = chỉ chạy bước
  sửa đúng một lần, không có vòng lặp phá/sửa.

### 7.7 Khi có ĐIỂM MỚI và khi có thay đổi khác

- Điểm mới xuất hiện: giữ nguyên các tuyến đang bay, chỉ chèn điểm mới vào chỗ rẻ nhất
  (đánh giá bằng bản đồ rủi ro belief TẠI THỜI ĐIỂM ĐÓ). Các điểm cũ không bị đụng tới.
- Vùng nguy hiểm đổi / deadline đổi / pin tụt: greedy KHÔNG lập lại kế hoạch (nó chỉ
  kích hoạt khi có điểm mới; src/baseline_greedy.py:22-23). UAV cứ bay theo tuyến cũ,
  kể cả khi tuyến đó giờ đi qua vùng nguy hiểm.
- Phản xạ an toàn (chung cho MỌI thuật toán, không phải lập kế hoạch): ngay trước khi
  bay tới điểm kế tiếp, nếu pin không đủ để tới đó rồi về depot thì bỏ điểm đó.
- Lưu ý nhỏ: khi greedy được kích hoạt vì có điểm mới, nó tính lại theo bản đồ rủi ro
  hiện tại, nên chặng nào giờ không còn khả thi sẽ bị cắt. Tức là greedy chỉ "tình cờ"
  phản ứng với rủi ro khi có điểm mới đến.

---

## 8. Belief và truth khác nhau thế nào?

Hỏi ngày 2026-11-17.
Code: src/risk_map.py (dòng 1-16 và hàm _state dòng 105), src/rolling_horizon.py
(dòng 37, 98, 216). Tài liệu: docs/mo_hinh_toan.md mục 1.2.

Cả hai đều là BẢN ĐỒ RỦI RO (vùng nguy hiểm ở đâu, nguy hiểm bao nhiêu, theo thời gian).

  - TRUTH  = thế giới THẬT: vùng nguy hiểm thực sự ở đâu, đi đâu, khi nào xuất hiện,
             rẽ hướng, biến mất. Chỉ "trọng tài" (bộ mô phỏng) biết.
  - BELIEF = những gì BỘ ĐIỀU PHỐI BIẾT tại thời điểm hiện tại t. Thuật toán chỉ được
             nhìn bản đồ này khi lập kế hoạch.

Ví dụ: dự báo thời tiết. Belief là bản dự báo sáng nay ("cơn bão đang đi về hướng đông,
tốc độ 20 km/h"). Truth là cơn bão thực sự: buổi trưa nó bất ngờ rẽ lên phía bắc.

### Ba chỗ belief khác truth

  | Điều                       | TRUTH (thật)               | BELIEF tại thời điểm t          |
  |----------------------------|----------------------------|---------------------------------|
  | Vùng nguy hiểm CHƯA xuất   | có, sẽ xuất hiện lúc sau   | KHÔNG BIẾT (không có trên bản   |
  | hiện                       |                            | đồ)                             |
  | Đường đi trong tương lai   | đi theo quỹ đạo thật, có   | đoán bằng cách KÉO DÀI THẲNG    |
  |                            | thể RẼ HƯỚNG               | theo vận tốc hiện tại; không    |
  |                            |                            | biết trước chỗ rẽ               |
  | Lúc biến mất               | biến mất ở thời điểm thật  | KHÔNG BIẾT, giả sử tồn tại mãi  |

Quá khứ và hiện tại thì belief khớp truth: vùng nguy hiểm đang ở đâu thì bộ điều phối
biết đúng chỗ đó. Chỉ TƯƠNG LAI là phải đoán.

### Ví dụ cụ thể

  Lúc t = 100, vùng nguy hiểm H ở (50,50), đang đi sang phải 0,1 đơn vị/thời gian.
  Thật ra lúc t = 150 nó rẽ lên trên.

  Belief lúc t = 100 đoán:  t = 200 -> H ở (60,50)    (kéo dài thẳng)
  Truth:                    t = 200 -> H ở (55,55)    (đã rẽ lúc 150)

  Nếu lúc t = 100 bộ điều phối xếp một UAV bay qua (55,55) lúc t = 200, nó nghĩ chỗ đó
  an toàn, nhưng thật ra UAV sẽ bay thẳng vào vùng nguy hiểm.

### Dùng ở đâu trong code?

  - LẬP KẾ HOẠCH (mọi thuật toán): dùng BELIEF tại thời điểm lập kế hoạch
    (build_problem gọi sim.truth.belief(t), src/rolling_horizon.py:216). Toàn bộ chi phí
    rủi ro và vùng cấm khi chọn tuyến đều tính trên belief.
  - CHẤM ĐIỂM (tính J, J_ref, ban_dist): UAV "bay thật" trên TRUTH
    (src/rolling_horizon.py:98). Rủi ro trong J là rủi ro UAV thực sự gặp.

Vì vậy một kế hoạch "trông an toàn" trên belief vẫn có thể bị chấm rủi ro cao trên truth.

### Vì sao chênh lệch này quan trọng?

Chênh lệch truth - belief chính là "BẤT NGỜ" mà thuật toán động phải xử lý.

  - greedy, alns_static: lập kế hoạch trên belief cũ, không cập nhật khi vùng nguy hiểm
    rẽ hướng, nên chịu trọn bất ngờ.
  - resolve, alns_dynamic: khi có sự kiện (vùng nguy hiểm mới, rẽ hướng), lấy BELIEF MỚI
    (đã thấy chỗ rẽ) rồi lập lại kế hoạch, nên né được.
  - alns_dynamic còn lập lại định kỳ mỗi 100, nên belief luôn "tươi" hơn.

Nếu belief = truth (biết trước tương lai) thì một kế hoạch tĩnh tốt là đủ, không cần
lập lại kế hoạch. Thiết kế belief/truth làm cho bài toán "động" thật sự và công bằng:
không thuật toán nào được nhìn trước tương lai.

---

## 9. Hotspot là gì? theta_ban là gì?

Hỏi ngày 2026-11-17.
Code: src/risk_map.py (Hotspot dòng 30, generate_hotspots dòng 176),
src/event_generator.py:23, config.yaml:8-9. Tài liệu: docs/mo_hinh_toan.md mục 1.2, C3.

### 9.1 Hotspot

Hotspot = MỘT VÙNG NGUY HIỂM DI ĐỘNG trên bản đồ: ví dụ cơn bão, đám cháy, vùng bị gây
nhiễu sóng... Bản đồ rủi ro được tạo bằng cách cộng gộp vài hotspot.

Mỗi hotspot có:
  | Thuộc tính     | Ý nghĩa                                         | Giá trị khi sinh   |
  |----------------|-------------------------------------------------|--------------------|
  | tâm c(t)       | vị trí tâm, di chuyển theo đường thẳng từng đoạn| ngẫu nhiên         |
  | sigma (σ)      | độ rộng: σ lớn thì vùng nguy hiểm lan rộng      | 5 - 8 đơn vị       |
  | amp (a)        | cường độ ở tâm (rủi ro tối đa, từ 0 đến 1)      | 0,85 - 1,0         |
  | t_start, t_end | lúc xuất hiện và lúc biến mất                   |                    |

Rủi ro của một hotspot tại điểm x giảm dần khi ra xa tâm, theo hình "quả chuông"
(Gauss):

  r(x,t) = a x exp( - khoảng_cách(x, tâm)^2 / (2 σ^2) )

  - Ngay tại tâm: r = a (gần 1, rất nguy hiểm).
  - Cách tâm 1σ:  r ≈ 0,61 a.
  - Cách tâm 2σ:  r ≈ 0,14 a.
  - Cách tâm 3σ:  r ≈ 0,01 a (gần như an toàn).

Nhiều hotspot chồng nhau thì gộp theo kiểu "noisy-OR":
  R = 1 - (1 - r_1)(1 - r_2)...
Hiểu là: xác suất "an toàn" = tích các xác suất an toàn của từng hotspot. Nhờ vậy R luôn
nằm trong [0, 1].

Ba loại hotspot (số lượng ở mức cường độ "med", src/event_generator.py:23-26):
  - Hotspot ban đầu (3 cái): có từ t = 0, bộ điều phối biết.
  - Hotspot rẽ hướng (2 lần): một hotspot ban đầu đột ngột đổi hướng/tốc độ (bất ngờ).
  - Hotspot mới (2 cái): xuất hiện giữa ngày, gần một điểm khách hàng; rủi ro tăng dần
    từ 40% lên 100% trong 150 đơn vị thời gian; tồn tại 250 - 450 rồi biến mất.
  Mức "low"/"high" có ít/nhiều sự kiện hơn.

Mỗi lần hotspot xuất hiện hoặc rẽ hướng là một sự kiện "risk_change": resolve và
alns_dynamic sẽ lập lại kế hoạch.

### 9.2 theta_ban (θ_ban) - ngưỡng cấm bay

theta_ban = MỨC RỦI RO TỐI ĐA được phép bay qua. Mặc định 0,8 (config.yaml:8).

  - Chỗ nào rủi ro R > 0,8 -> VÙNG CẤM: tuyến bay không được đi qua, dù chỉ một điểm
    trên chặng (ràng buộc cứng C3).
  - Chỗ nào R ≤ 0,8 -> ĐƯỢC bay qua, nhưng bị tính phạt rủi ro (5 x phơi nhiễm trong J).

Tức là rủi ro được xử lý hai tầng:
  rủi ro thấp/vừa -> phạt mềm (đi qua được, nhưng tốn chi phí)
  rủi ro rất cao  -> cấm cứng (không được đi qua)

Ngoại lệ: trong bán kính 8 đơn vị quanh depot (depot_safe, config.yaml:9) không áp lệnh
cấm, để UAV luôn cất/hạ cánh được.

Kiểm tra lúc lập kế hoạch dùng BELIEF; sau khi bay, quãng đường thực sự bay trong vùng
R > theta_ban trên TRUTH được báo cáo là ban_dist (đo mức "vi phạm do bất ngờ").

### 9.3 Vùng cấm to cỡ nào?

Với một hotspot, vùng cấm là hình tròn quanh tâm, bán kính:
  d = σ x căn( 2 x ln(a / theta_ban) )   (chỉ tồn tại khi a > theta_ban)

  Ví dụ σ = 6, a = 0,9:
    theta_ban = 0,8 -> bán kính ≈ 2,9 đơn vị  (chỉ phần lõi bị cấm)
    theta_ban = 0,6 -> bán kính ≈ 5,4 đơn vị  (vùng cấm rộng gần gấp đôi)
  Nếu a = 0,85 < 0,9 thì với theta_ban = 0,9 hotspot đó KHÔNG có vùng cấm (chỉ phạt).
  Hai hotspot chồng nhau có thể tạo vùng cấm lớn hơn (do noisy-OR).

### 9.4 theta_ban ảnh hưởng kết quả thế nào?

Độ nhạy với alns_dynamic (tables/sensitivity.md:16-19; docs/ket_qua_tom_tat.md:9):

  | theta_ban | J_ref | quãng đường | trễ   | bỏ sót (TB) |
  |-----------|-------|-------------|-------|-------------|
  | 0,6       | 2279  | 549         | 410   | 3,25        |
  | 0,7       | 1843  | 542         | 326   | 2,00        |
  | 0,8       | 1515  | 504         | 199   | 1,50        |
  | 0,9       | 1378  | 471         | 213   | 1,17        |

  theta_ban NHỎ = cấm khắt khe hơn = vùng cấm rộng hơn -> phải bay vòng xa hơn, trễ hơn,
  bỏ nhiều điểm hơn (mỗi điểm phạt 300) -> J tăng mạnh. Đây là tham số NHẠY NHẤT.

  Lưu ý khi viết báo cáo: J thấp hơn ở theta_ban = 0,9 KHÔNG có nghĩa 0,9 "tốt hơn".
  Nó chỉ nghĩa là cho phép bay gần vùng nguy hiểm hơn. Chọn theta_ban là quyết định về
  mức an toàn chấp nhận được, không phải chỉ để tối ưu J.

---

## 10. Vòng lặp phá - sửa trong ALNS; vì sao greedy = "sửa một lần"?

Hỏi ngày 2026-11-17 (nối tiếp mục 4 và mục 7).
Code: src/alns_dynamic.py (initial_solution dòng 57, solve dòng 66, vòng lặp dòng 89-135),
src/destroy_ops.py, src/repair_ops.py.

### 10.1 Hai thao tác cơ bản

  PHÁ (destroy): GỠ một số điểm ra khỏi tuyến. Các điểm này thành "chưa xếp".
  SỬA (repair):  CHÈN các điểm "chưa xếp" vào tuyến (cách chèn: mục 7 - tìm chỗ rẻ
                 nhất, chèn từng điểm một).

Thao tác SỬA không quan tâm điểm "chưa xếp" từ đâu ra: do vừa bị PHÁ gỡ ra, hay do tuyến
đang rỗng từ đầu. Nó chỉ việc chèn tất cả vào.

### 10.2 ALNS chạy thế nào? (toàn bộ quy trình) 

  Bước 0 - Phương án ban đầu: bắt đầu từ tuyến rỗng (hoặc kế hoạch cũ nếu warm-start),
           chạy SỬA một lần để chèn mọi điểm -> có phương án đầu tiên.
  Bước 1..N - Lặp N vòng (N = 150/250/400/600 tùy cỡ bài):
       a. Sao chép phương án hiện tại.
       b. PHÁ: gỡ q điểm (q = 10-30% số điểm đã xếp, tối đa 30).
       c. SỬA: chèn lại q điểm đó.
       d. So sánh J mới với J hiện tại:
            - tốt hơn kỷ lục          -> nhận, lưu làm kỷ lục
            - tốt hơn hiện tại        -> nhận
            - kém hơn                 -> thỉnh thoảng vẫn nhận (simulated annealing)
            - còn lại                 -> bỏ, giữ phương án cũ
  Kết quả: phương án KỶ LỤC (J nhỏ nhất từng thấy).

### 10.3 Ví dụ một vòng (số liệu giả định)

  Phương án hiện tại (J = 120):
    UAV 1: depot -> A -> B -> C -> depot
    UAV 2: depot -> D -> E -> F -> depot

  PHÁ (toán tử "related": gỡ các điểm gần nhau) -> gỡ C và D
    UAV 1: depot -> A -> B -> depot
    UAV 2: depot -> E -> F -> depot
    Chưa xếp: {C, D}

  SỬA (greedy) -> chèn lại C, D vào chỗ rẻ nhất, có thể KHÁC chỗ cũ:
    UAV 1: depot -> A -> D -> B -> depot
    UAV 2: depot -> C -> E -> F -> depot
    J mới = 112 < 120  -> NHẬN.

  Vòng sau lại gỡ vài điểm khác, xếp lại... sau hàng trăm vòng, J giảm dần.

Mấu chốt: lúc sửa, các điểm còn lại đã khác (vì một số điểm đã bị gỡ), nên "chỗ rẻ nhất"
cũng khác. Gỡ ra rồi chèn lại là cách để SỬA SAI những lựa chọn chèn trước đó.

### 10.4 Vì sao nói greedy = "chỉ chạy bước sửa một lần"?

  ALNS   = Bước 0 (SỬA một lần từ tuyến rỗng)  +  N vòng (PHÁ -> SỬA -> so sánh)
  greedy = Bước 0 (SỬA một lần từ tuyến rỗng)  +  KHÔNG có vòng nào

Tức greedy dừng ngay sau khi có phương án đầu tiên. Thủ tục chèn của greedy chính là
toán tử sửa "greedy" của ALNS (src/baseline_greedy.py gọi repair_greedy).

Hệ quả: lựa chọn chèn sai ở đầu (ví dụ điểm C bị nhét vào UAV 1 trong khi đặt vào UAV 2
rẻ hơn, nhưng lúc chèn C thì chưa thấy) sẽ không bao giờ được sửa. ALNS thì gỡ C ra ở
một vòng nào đó và chèn lại đúng chỗ.

Khi có điểm mới, greedy cũng chỉ chạy SỬA cho riêng điểm mới đó; không PHÁ gì cả.

### 10.5 Liên hệ các thuật toán

  | Thuật toán   | Bước 0 (sửa ban đầu)     | Vòng phá-sửa                  |
  |--------------|--------------------------|-------------------------------|
  | greedy       | sửa greedy               | không có                      |
  | alns_static  | sửa regret-2             | 4N vòng, chỉ một lần lúc t=0   |
  | resolve      | sửa regret-2 (từ rỗng)   | N vòng, mỗi lần có sự kiện     |
  | alns_dynamic | từ kế hoạch cũ (warm)    | N vòng, mỗi lần có sự kiện và  |
  |              | + sửa regret-2 điểm mới  | mỗi 100; có toán tử rủi ro     |

---

## 11. Regret-2 là gì?

Hỏi ngày 2026-11-17.
Code: src/repair_ops.py:136-173 (_greedy_like với regret=True, repair_regret2).

Regret-2 ("hối tiếc bậc 2") là một cách SỬA (chèn điểm), khác greedy ở THỨ TỰ chọn điểm
nào chèn trước.

### 11.1 Greedy chọn thế nào? (nhắc lại mục 7)

  Chèn trước điểm có chi phí chèn RẺ NHẤT.
  -> Điểm dễ được chèn trước, điểm khó bị để lại sau, lúc đó chỗ tốt đã bị chiếm hết.

### 11.2 Regret-2 chọn thế nào?

Với mỗi điểm chưa xếp, tính chi phí chèn tốt nhất trên TỪNG UAV, rồi lấy:
  - best1 = chi phí chèn rẻ nhất (trên UAV tốt nhất)
  - best2 = chi phí chèn rẻ thứ hai (trên UAV tốt thứ hai)

  regret = best2 - best1
         = "nếu bây giờ không chèn, chỗ tốt nhất bị chiếm mất, thì sẽ thiệt thêm bao nhiêu"

Chèn trước điểm có REGRET LỚN NHẤT (nếu bằng nhau thì chọn điểm có best1 nhỏ hơn).
Điểm chỉ chèn được vào đúng MỘT UAV được ưu tiên cao nhất (vì nếu UAV đó đầy là mất).

Số "2" nghĩa là so sánh lựa chọn tốt nhất với lựa chọn tốt THỨ 2.

### 11.3 Ví dụ (số liệu giả định)

  Hai điểm X, Y chưa xếp; hai UAV.

  | Điểm | Chèn vào UAV1 | Chèn vào UAV2 | best1 | regret = best2 - best1 |
  |------|---------------|---------------|-------|------------------------|
  | X    | 10            | 12            | 10    | 2                      |
  | Y    | 15            | 60            | 15    | 45                     |

  Greedy:    chèn X trước (10 < 15). Giả sử X chiếm chỗ cuối cùng của UAV1 (hết tải),
             Y buộc phải sang UAV2 với giá 60. Tổng = 10 + 60 = 70.
  Regret-2:  chèn Y trước (regret 45 > 2) vào UAV1 giá 15. X sang UAV2 giá 12.
             Tổng = 15 + 12 = 27.

  X có lựa chọn thay thế gần như tốt ngang nhau, nên để sau cũng không sao. Y thì không,
  nên phải "giữ chỗ" cho Y trước.

Ví dụ đời thường: xếp chỗ ngồi. Người nào chỉ ngồi được đúng một ghế thì xếp trước;
người ngồi đâu cũng được thì xếp sau.

### 11.4 Dùng ở đâu trong dự án?

  - Tạo phương án ban đầu của ALNS (bước 0, src/alns_dynamic.py:57-63): luôn dùng
    regret-2.
  - Là một trong các toán tử SỬA trong vòng lặp ALNS: greedy, regret2 (và riskprio ở
    alns_dynamic). Mỗi vòng chọn một toán tử sửa (ngẫu nhiên, hoặc theo trọng số thích
    nghi).
  - greedy (thuật toán baseline) KHÔNG dùng regret-2, chỉ dùng chèn rẻ nhất.

Trong code, cả greedy và regret-2 đều tính chi phí chèn của mọi điểm trên mọi UAV;
chúng chỉ khác ở luật chọn điểm nào chèn trước. Regret-2 thường có lợi khi tải/pin chặt
hoặc cửa sổ thời gian hẹp (nhiều điểm chỉ có ít chỗ chèn được).

---

## 12. Các thuật toán trong bài Pu et al. (2026) - docs/1.pdf

Hỏi ngày 2026-11-17 (yêu cầu chạy lại thí nghiệm của bài báo).
Mã tái lập: thư mục paper_repro/. Báo cáo so sánh: docs/tai_lap_pu2026.md.

Bức tranh chung: bài chia việc thành 2 tầng + 1 bộ dự báo.

  task (x, y, loại) --> [3D-SOM: chia nhóm] --> [Đấu giá: tinh chỉnh] --> [TSP phân tầng: xếp thứ tự bay]
                                                       ^
  chuỗi nhu cầu vật tư --> [RBF-MKELM: dự báo] --------+ (dự báo >= 90 -> sinh task tiếp tế)

Hàm chi phí dùng để chấm mọi thuật toán (Eq.14, paper_repro/common.py:cost_from):

  C = 1 x P + 1,5 x D + 2 x R
  P = tổng quãng đường 5 UAV ; D = độ lệch chuẩn quãng đường (cân bằng tải) ;
  R = độ lệch chuẩn tổng điểm task (cân bằng loại việc).
  Lưu ý: bài viết D, R chia cho u (độ lệch chuẩn tổng thể), nhưng ví dụ số ở Hình 4
  (std của 20 và 15 = 3,533) là độ lệch chuẩn MẪU (chia u-1, mặc định của MATLAB).
  Ta dùng u-1 cho khớp số của bài.

### 12.1 SOM / 3D-SOM (Self-Organizing Map) - paper_repro/som3d.py

  SOM là mạng nơ-ron học KHÔNG giám sát dùng để gom cụm. Mỗi neuron có một vector
  trọng số (một "tâm"). Lặp nhiều lần: lấy 1 điểm, tìm neuron gần nhất (gọi là BMU -
  Best Matching Unit), kéo BMU và các neuron "hàng xóm" của nó về phía điểm đó.
  Mức kéo (learning rate) và bán kính hàng xóm giảm dần theo vòng lặp.
  Khác K-means ở chỗ: neuron hàng xóm cũng bị kéo theo -> các cụm kề nhau có trật tự.

  "3D" trong bài: mỗi task là (x, y, loại), chuẩn hóa về [0,1]. Khoảng cách =
  khoảng cách không gian + alpha x |chênh lệch loại|. Số neuron = số UAV.
  alpha = 0 -> chỉ gom theo vị trí; alpha = 3 -> ưu tiên gom task CÙNG LOẠI cho một UAV.

  Tái lập cho thấy: alpha = 3 gom theo loại rất mạnh, có khi 2/5 UAV không nhận task
  nào ("neuron chết"). Gom theo loại cũng làm mỗi UAV phải bay khắp bản đồ.

### 12.2 Đấu giá (auction) - paper_repro/auction.py

  Thực chất là tìm kiếm cục bộ "di chuyển một task": với từng task, thử chuyển nó
  sang từng UAV khác; nếu chi phí Eq.14 giảm thì giữ. Lặp tối đa 20 vòng.
  Gọi là "đấu giá" vì mỗi UAV như đang "trả giá" cho task; nhưng cách tính trong bài
  là đánh giá chi phí TOÀN CỤC, nên giống thuật toán leo đồi tập trung hơn là đấu giá
  phân tán thật sự.

### 12.3 TSP, NN, 2-Opt, TSP phân tầng - paper_repro/common.py

  TSP: tìm thứ tự đi qua tất cả điểm sao cho quãng đường ngắn nhất.
  NN (nearest neighbour): luôn bay tới điểm gần nhất chưa thăm. Nhanh nhưng hay bỏ
    sót điểm ở xa, sau đó phải quay lại.
  2-Opt: lấy 2 cạnh của lộ trình, đảo ngược đoạn giữa; nếu ngắn hơn thì giữ. Lặp
    đến khi không cải thiện được -> gỡ các chỗ đường bay cắt chéo nhau.
  TSP phân tầng: chia task theo loại ưu tiên A > B > C; bay hết A (tối ưu bằng
    NN+2-Opt), từ điểm A cuối cùng bay sang tập B, rồi C. Không quay về base.

### 12.4 Các thuật toán so sánh (baseline) - paper_repro/baselines.py

  Mọi baseline đều tìm "UAV nào nhận task nào" (vector gán), rồi chấm bằng Eq.14.
  GA (di truyền): quần thể lời giải; lai ghép 2 lời giải cha mẹ + đột biến; giữ con tốt.
  PSO (bầy đàn): mỗi "hạt" là một lời giải, di chuyển về phía tốt nhất của nó và của cả bầy.
  GA-PSO: PSO, nhưng mỗi vòng thêm lai ghép/đột biến của GA.
  NSGA-II: GA đa mục tiêu; giữ các lời giải "không bị trội" (Pareto) trên 3 mục
    tiêu P, D, R; cuối cùng chọn lời giải có Eq.14 nhỏ nhất.
  ACO (đàn kiến): mỗi kiến gán task theo xác suất tỉ lệ với "pheromone" (kinh nghiệm
    các vòng trước) và độ gần; lời giải tốt rải nhiều pheromone hơn.
  DQN (deep Q-learning): mạng nơ-ron học "gán task này cho UAV nào thì lợi" qua
    200 tập thử-sai. Ta viết bằng numpy (không dùng PyTorch).
  NN: gán tham lam, không ngẫu nhiên -> chỉ chạy 1 lần.
  KCA: K-means gom cụm theo vị trí rồi đấu giá. Giống OUR nhưng thay 3D-SOM bằng K-means.

### 12.5 Dự báo: RBF, ELM, KELM, MKELM, RBF-MKELM - paper_repro/forecast.py

  Bài toán: biết nhu cầu 3 bước trước, đoán nhu cầu bước kế tiếp.
  RBF network: mỗi neuron ẩn là một "chuông Gauss" đặt tại một tâm; đầu ra là tổng
    có trọng số của các chuông.
  ELM (Extreme Learning Machine): mạng 1 lớp ẩn, trọng số lớp ẩn chọn NGẪU NHIÊN và
    không huấn luyện; chỉ giải trọng số đầu ra bằng bình phương tối thiểu -> rất nhanh.
  KELM: thay lớp ẩn ngẫu nhiên bằng một hàm kernel (độ giống nhau giữa 2 mẫu); giống
    hồi quy ridge dùng kernel. C là tham số chống quá khớp (C lớn = ít chống).
  MKELM: KELM với kernel là TỔ HỢP nhiều kernel (bài dùng đa thức + "wavelet Morlet").
  RBF-MKELM (đề xuất của bài): biến 3 giá trị lịch sử thành 15 đặc trưng RBF (15
    chuông Gauss, tâm tìm bằng K-means), rồi đưa vào MKELM. Có 7 siêu tham số
    (delta, C, a, b, c, d, alpha), dò tự động bằng AOO.
  ARIMA: mô hình thống kê cổ điển cho chuỗi thời gian (tự hồi quy + sai phân + trung bình trượt).
  RF (Random Forest): trung bình của 100 cây quyết định.
  MSE / MAE / R2: sai số bình phương TB / sai số tuyệt đối TB / mức giải thích phương sai
    (1 là hoàn hảo).

  Điểm cần nhớ khi đọc Bảng 7: dữ liệu có nhiễu 2 x N(0,1), phương sai 4.
  Vì vậy mô hình tốt mấy thì MSE cũng không xuống dưới ~4 một cách bền vững.
  Các MSE 3-5 trong bài và của ta đều đã ở "sàn nhiễu".

### 12.6 AOO (Animated Oat Optimization) - paper_repro/aoo.py

  Thuật toán tối ưu "lấy cảm hứng từ thiên nhiên" (2025), mô phỏng hạt yến mạch:
  phát tán xa (khám phá ngẫu nhiên), lăn khi hút ẩm (tìm quanh lời giải tốt nhất),
  bật văng khi gặp vật cản (nhảy khỏi cực tiểu địa phương), kèm bước Lévy.
  Không đọc được bài gốc; bản trong paper_repro là dựng lại từ mô tả trong bài
  RLDN-AOO (Electronic Research Archive 2025), một số hệ số là ước đoán.
  Một bài KBS 2026 ("Critical analysis of the animated oat optimization algorithm")
  đã phê bình AOO. Ta chạy thêm bản thay AOO bằng Differential Evolution (DE) để đối chiếu.

  Bẫy quan trọng: bài dò siêu tham số bằng RMSE trên CHÍNH tập train. Với kernel
  linh hoạt, tối ưu kiểu đó cho RMSE train ~0 nhưng dự báo tập kiểm định rất tệ
  (quá khớp). Ta tái lập đúng cách đó và MSE kiểm định lên tới hàng triệu; phải tách
  phần holdout trong tập train thì mới có MSE ~6.

---

## 13. VRPTW / dVRPTW là gì? Instance Solomon là gì?

Hỏi ngày 2026-11-17 (mục 1 "Tổng quan" của tài liệu tổng hợp).

VRP = Vehicle Routing Problem, tức bài toán định tuyến xe: có một kho (depot) và nhiều
khách hàng. Phải chia khách cho các xe và xếp thứ tự ghé thăm sao cho tổng chi phí nhỏ nhất.

  TW (Time Windows), tức cửa sổ thời gian: mỗi khách chỉ nhận hàng trong khoảng [mở, đóng].
     Đến sớm thì phải chờ, đến muộn thì bị tính trễ (số hạng w_l trong J).
  d (dynamic), tức động: không biết trước toàn bộ dữ liệu. Khách mới xuất hiện, deadline
     đổi, pin tụt trong lúc đang bay. Ngược lại là bản "tĩnh", biết hết từ t = 0.

Vậy dVRPTW = định tuyến xe có cửa sổ thời gian, dữ liệu thay đổi theo thời gian. Ở đây
"xe" là UAV, và ta cộng thêm chi phí phơi nhiễm rủi ro (w_r) vào J (docs/mo_hinh_toan.md:65).

Instance Solomon: bộ đề chuẩn của VRPTW (Solomon, 1987). Ai nghiên cứu VRPTW cũng thử trên
bộ này để kết quả so sánh được với nhau. Có 3 nhóm:
  C  = khách tụm thành cụm,
  R  = khách rải ngẫu nhiên,
  RC = trộn cả hai.
Tên file như C101_050 nghĩa là bài C101, lấy 50 khách đầu (config.yaml:23-38).
Gehring-Homberger là bộ đề mở rộng cùng kiểu, có 200 khách trở lên (dùng ở thí nghiệm "large").

---

## 14. Ablation và độ nhạy là gì?

Hỏi ngày 2026-11-17.

Ablation (nghĩa gốc là "cắt bỏ"): lần lượt TẮT từng bộ phận của thuật toán, giữ nguyên
mọi thứ khác, rồi xem J tăng bao nhiêu. Bộ phận nào tắt đi làm J tăng nhiều thì bộ phận
đó quan trọng. Các biến thể nằm ở config.yaml:52:
  dyn_no_rolling   bỏ rolling-horizon            -> J +20,3%
  resolve_periodic giải lại "nguội" mỗi chu kỳ   -> J +12,8%
  dyn_no_warm      bỏ warm-start                 -> J +10,7%
  dyn_no_riskprio  bỏ toán tử risk-priority      -> J +0,2%  (gần như không đổi)
  dyn_no_riskrem   bỏ toán tử risk-based removal -> J -0,4%  (gần như không đổi)
  (docs/ket_qua_tom_tat.md:7)
Kết luận: rolling và warm-start có tác dụng rõ. Hai toán tử "nhận biết rủi ro" thì chưa
cho thấy đóng góp đo được.

Độ nhạy (sensitivity analysis): GIỮ thuật toán nguyên vẹn, nhưng thay đổi một THAM SỐ
qua nhiều mức (ví dụ w_r = 0; 2,5; 5; 10; 20), rồi xem J thay đổi thế nào (config.yaml:56-63).
Mục đích là trả lời hai câu hỏi: kết quả có phụ thuộc mạnh vào việc chọn tham số không,
và tham số nào cần chỉnh cẩn thận. Ở đây theta_ban nhạy nhất; q_hi, số vòng, dt lệch < 5%.

So sánh ngắn: ablation thay đổi CẤU TRÚC thuật toán (bật/tắt bộ phận), còn độ nhạy
thay đổi GIÁ TRỊ tham số.

---

## 15. Kiểm định thống kê: Friedman, Wilcoxon, Holm, p_holm, A12

Hỏi ngày 2026-11-17. Code: stats_tests.py; kết quả: tables/stats_tests.md.

Vấn đề: ALNS có yếu tố ngẫu nhiên, nên chạy lần khác sẽ ra J khác. Câu hỏi đặt ra là
alns_dynamic thắng thật, hay chỉ thắng nhờ may mắn? Kiểm định thống kê trả lời câu đó.

Ghép cặp: mỗi (instance, seed) là một "khối" có cùng kịch bản cho cả 4 thuật toán.
Có 12 instance x 20 seed = 240 khối, nên so sánh rất công bằng.

  Friedman: kiểm tra chung "4 thuật toán có như nhau không?". Trong mỗi khối, xếp hạng
    thuật toán từ 1 (J thấp nhất) đến 4. Nếu 4 thuật toán như nhau thì hạng trung bình
    phải gần bằng nhau. Thực tế: dynamic 1,34; resolve 2,10; static 2,75; greedy 3,81.
    p = 5e-102, gần như không thể là ngẫu nhiên.
  Wilcoxon (signed-rank): so từng CẶP, ví dụ dynamic với resolve trên cùng 240 khối,
    xem chênh lệch nghiêng hẳn về một phía hay không.
  Holm: so 3 cặp cùng lúc thì khả năng "trúng may" tăng lên. Holm nâng p lên để bù cho
    việc so nhiều lần. p_holm là p SAU khi đã hiệu chỉnh.
  p-value: xác suất thấy chênh lệch lớn như vậy NẾU thật ra hai thuật toán như nhau.
    p < 0,05 thì coi là "có ý nghĩa". p_holm < 1e-22 là cực nhỏ. Còn ở 200 khách,
    p = 0,21 là lớn: chưa đủ bằng chứng để nói dynamic hơn resolve.
  A12 / Cliff delta (độ lớn hiệu ứng): p nhỏ chỉ cho biết "có khác", không cho biết
    "khác nhiều hay ít". A12 = xác suất một lần chạy của dynamic có J thấp hơn đối thủ.
    A12 = 0,5 nghĩa là ngang nhau. So với resolve, A12 = 0,55: dynamic thắng chắc chắn,
    nhưng chỉ nhỉnh hơn một chút (hiệu ứng nhỏ). So với greedy, A12 = 0,70: hơn rõ.

---

## 16. Bảng giả thuyết H1-H4 đọc thế nào?

Hỏi ngày 2026-11-17. Nguồn: docs/idea.md, docs/ket_qua_tom_tat.md.

Giả thuyết là điều ta ĐOÁN trước khi chạy thí nghiệm. Cột cuối của bảng ghi kết quả có
khớp với điều đoán hay không.

  H1: dynamic tốt hơn 3 đối thủ, và có ý nghĩa thống kê (mục 15).
      Đạt với instance <= 100 khách. Ở 200 khách dynamic vẫn thấp hơn resolve 2,9%,
      nhưng p = 0,21 nên chưa khẳng định được (chỉ có 5 seed, thắng 8/15 cặp).
  H2: bỏ warm-start hoặc bỏ rolling thì J tăng. Đạt (+10,7% và +20,3%, xem mục 14).
  H3: sự kiện càng dày (cường độ low -> high) thì dynamic càng có lợi so với đối thủ.
      "Cường độ" = số hotspot mới, số lần hotspot đổi hướng, số deadline đổi, số lần
      pin tụt (src/event_generator.py:22-26; ví dụ low đổi 2 deadline, high đổi 8).
      Hiện mới biết J của dynamic tăng từ 1322 lên 1563 khi cường độ tăng, nhưng CHƯA đo
      KHOẢNG CÁCH giữa các thuật toán theo cường độ. Vì vậy H3 chưa kiểm được.
  H4: tăng w_r (trọng số rủi ro) thì có lợi đến một mức, sau đó không lợi thêm.
      "Dạng chữ U" (ở <= 50 khách): J cao khi w_r = 0 (bay liều qua vùng nguy hiểm),
      thấp nhất ở w_r = 5-10, rồi cao lại ở w_r = 20 (né rủi ro quá mức, đi vòng xa và
      bỏ rơi khách). Ở 200 khách thì J tăng đều theo w_r, không có đáy chữ U: rủi ro
      giảm từ 251 xuống 170, nhưng trễ tăng từ 368 lên 1054. Đó là một sự đánh đổi.

---

## 17. Đa mục tiêu, NSGA-III, HV và IGD

Hỏi ngày 2026-11-17. Code: run_mo.py, src/nsga3.py; kết quả: tables/mo_static.md.

Đa mục tiêu: J cộng 4 thứ lại với trọng số cố định (w_d, w_r, w_l, w_u). Nếu KHÔNG cộng
gộp mà giữ riêng 3 mục tiêu (quãng đường, rủi ro, trễ), thì không có một lời giải "tốt
nhất" duy nhất. Thay vào đó có một TẬP lời giải mà ở mỗi lời giải, muốn cải thiện mục tiêu
này thì phải chịu thiệt ở mục tiêu khác. Tập đó gọi là mặt Pareto (Pareto front).
Ví dụ: đường ngắn nhưng qua vùng nguy hiểm, so với đường an toàn nhưng dài hơn.

NSGA-III: thuật toán di truyền (genetic algorithm) chuyên tìm cả mặt Pareto cùng lúc.
  Nó là phương pháp chuẩn để so sánh trong các bài nhiều mục tiêu. Ở đây ta so với
  "ALNS archive", tức các lời giải không bị trội mà ALNS gặp trong quá trình chạy.

Hai thước đo chất lượng của một mặt Pareto:
  HV (Hypervolume): thể tích vùng mà mặt Pareto "phủ" được tính tới một điểm mốc xấu.
    CÀNG LỚN CÀNG TỐT, vì phủ rộng và sát lời giải tối ưu.
  IGD (Inverted Generational Distance): khoảng cách trung bình từ mặt Pareto tốt nhất
    đã biết tới mặt Pareto của thuật toán. CÀNG NHỎ CÀNG TỐT.
Kết quả: ALNS có HV 2,40e8 so với 0,81e8, IGD 947 so với 1426, nên ALNS hơn (p = 0,003).
Lưu ý: NSGA-III ở đây là bản tự viết và chưa tinh chỉnh, nên không được kết luận quá mạnh.
Phép so sánh này làm trên bài TĨNH, không phải bài động.

---

## 18. RescueNet và "mặt nạ phân đoạn"

Hỏi ngày 2026-11-17. Code: src/rescuenet_risk.py; danh sách lớp: docs/rescuenet_classes.md.

RescueNet là bộ ảnh chụp từ UAV sau một trận bão lớn. Mỗi ảnh đi kèm một
"mặt nạ phân đoạn" (segmentation mask): một ảnh cùng kích thước, trong đó mỗi điểm ảnh
được gán nhãn là nước, đường, nhà hư hỏng nặng, cây, v.v.
Ý tưởng: đổi nhãn thành mức rủi ro (ví dụ nhà sập thì rủi ro cao, đường trống thì thấp),
ta được một bản đồ rủi ro TĨNH dựa trên ảnh thật, rồi cộng nó vào các hotspot động.
Mức rủi ro gán cho từng lớp là do ta tự đặt ra, không có trong dữ liệu gốc. Vì vậy đây
mới là "hướng mở". Kết quả: hai bản động ngang nhau (dynamic 5566, resolve 5467).

---

## 19. "Tái hiện đơn giản, chỉ so sánh định tính" nghĩa là gì?

Hỏi ngày 2026-11-17.

Pu et al. (2026) làm cùng loại bài toán. Nhưng ta KHÔNG có code và dữ liệu của họ,
nên mô hình và bộ đề của ta là bản tự dựng, đơn giản hơn.
  Định lượng = so con số trực tiếp ("J của ta 2201, của họ 2500"). Ở đây KHÔNG làm được,
    vì hai bên đo trên hai bài toán khác nhau.
  Định tính = chỉ so xu hướng và kết luận ("cả hai đều thấy lập lại kế hoạch động tốt
    hơn kế hoạch tĩnh").
Khi viết báo cáo, tránh đặt số của mình cạnh số của họ như thể chúng so sánh được.
