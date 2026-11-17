# data/rescuenet
5 mặt nạ phân đoạn RescueNet (tập test) dùng cho `run_rescuenet.py`. Chỉ có nhãn, không có ảnh gốc.
Nguồn: RescueNet, Rahnemoonfar, Chowdhury, Murphy, Sci. Data 10:913 (2023), bản figshare chính thức. Giấy phép CC BY-NC-ND 4.0, chỉ dùng phi thương mại.
Chỉ số lớp 0–10 xem docs/rescuenet_classes.md (đã kiểm bằng np.unique trên mẫu).
Toàn bộ nhãn val (449) và test (450) có trong gói dữ liệu gốc; train chỉ có 2135/3595 file.

**Trên GitHub không kèm các file `*_lab.png`.** Muốn chạy `run_rescuenet.py` thì tải gói RescueNet từ nguồn gốc (figshare) và chép 5 mặt nạ test sau vào thư mục này: 10801, 10807, 10917, 11358, 14898 (`<id>_lab.png`).
