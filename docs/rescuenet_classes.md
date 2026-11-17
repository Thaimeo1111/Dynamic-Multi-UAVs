# RescueNet: lop va chi so nhan; mo ta C2A

## 1. RescueNet (Rahnemoonfar, Chowdhury, Murphy, Sci. Data 10:913, 2023)

Nguon chinh: muc Methods ("Generation of segmentation mask and classification label") cua bai bao, doc qua
- https://dx.doi.org/10.1038/s41597-023-02799-4 (ban Scientific Data), va
- https://arxiv.org/pdf/2202.12361 (ban arXiv).

Hai lan doc doc lap cho cung ket qua. Doan goc: "Water" = 1, "Building No Damage" = 2, "Building Minor Damage" = 3, "Building Major Damage" = 4, "Building Total Destruction" = 5, "Vehicle" = 6, "Road-Clear" = 7, "Road-Blocked" = 8, "Tree" = 9, "Pool" = 10; cac pixel con lai la "Background" = 0.

| Chi so (pixel value) | Lop |
|---|---|
| 0 | Background |
| 1 | Water |
| 2 | Building No Damage |
| 3 | Building Minor Damage (bai cung viet "Medium" o cho khac, cung mot lop) |
| 4 | Building Major Damage |
| 5 | Building Total Destruction |
| 6 | Vehicle |
| 7 | Road-Clear |
| 8 | Road-Blocked |
| 9 | Tree |
| 10 | Pool |

- Muc do chac chan: CAO ve thu tu/chi so theo van ban bai bao. Han che: README tren GitHub chinh thuc (https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation) khong doc duoc truc tiep (khong duoc phep truy cap tu moi truong nay), va chua kiem tra file mask thuc te. Truoc khi dung, nen chay `np.unique` tren vai mask PNG de xac nhan gia tri 0-10.
- Nhan phan loai cap anh la he khac (0 = Superficial, 1 = Medium, 2 = Major damage), khong nham voi chi so phan doan.
- Quy mo: 4.494 anh UAV, 3000x4000 pixel, 10 lop gan nhan + nen; chia 3.595 train / 449 val / 450 test (https://arxiv.org/pdf/2202.12361).
- Tai du lieu (theo ban arXiv v4): Dropbox https://www.dropbox.com/scl/fo/ntgeyhxe2mzd2wuh7he7x/AHJ-cNzQL-Eu04HS6bvBgcw?rlkey=6vxiaqve9gp6vzvzh3t5mz0vv&e=1&dl=0 (chua mo thu).
- Mot bai khac dung RescueNet (https://arxiv.org/pdf/2412.10756) liet ke cac lop nhung khong ghi chi so, nen khong dung lam nguon chi so.
- Goi y de gan voi risk map: lop co rui ro cao co the la 4, 5, 8 (damage lon, sap hoan toan, duong bi chan); day la de xuat cua nguoi lap tai lieu, KHONG phai thong tin tu nguon.

## 2. C2A (Nihal, Yen, Itoyama, Nakadai, arXiv:2408.04922; chap nhan ICPR 2024)

Nguon: https://arxiv.org/abs/2408.04922 va https://arxiv.org/html/2408.04922v2.

- Ten day du: "Combination to Application" (C2A), dataset phat hien nguoi cho tim kiem cuu nan, tao bang cach ghep tu the nguoi len nen anh UAV thien tai.
- Kich thuoc: 10.215 anh; hon 360.000 doi tuong nguoi (bounding box ngang).
- Nen anh lay tu AIDER voi 4 loai tham hoa: chay/khoi (320), lu lut (370), nha sap/dong do nat (320), tai nan giao thong (335); bo 1.200 anh binh thuong.
- Tu the nguoi lay tu LSP/MPII-MPHB (26.675 anh, 29.732 doi tuong), cat bang U2-Net, ngau nhien doi ti le/vi tri; 5 lop tu the: cui, quy, nam, ngoi, dung.
- Do phan giai goc khong dong nhat (khoang 123x152 den 5184x3456, bang 2 ghi 150-3400 khac voi van ban); chia train/val/test chua xac minh.
- URL du lieu theo bai: https://github.com/Ragib-Amin-Nihal/C2A (chua mo thu).
- Day la du lieu tong hop (ghep), khong phai anh thuc tu thien tai; neu dung nhu nguon nhu cau cuu ho trong bai thi can noi ro.
