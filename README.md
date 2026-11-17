# Dynamic risk-aware multi-UAV routing (dVRPTW, rolling-horizon ALNS)

## Chạy thí nghiệm
```bash
python3 -m pytest tests -q                 # 21 test (hoặc python3 -m unittest)
python3 prepare_real_data.py <thu_muc>     # cắt Solomon gốc 25/50/100 và Homberger 200 vào data/
python3 run_small.py                       # T08/T09: 3 baseline trên 6 instance nhỏ × 5 seed
python3 tune_alns.py                       # T24: chọn số vòng ALNS
./run_all_experiments.sh                   # main -> ablation -> large -> sensitivity -> sensitivity_large (có --resume)
python3 run_batch.py --exp main --seeds 3 --out results/raw_main.csv --resume   # chạy thử nhanh
python3 run_mo.py                          # T21: NSGA-III vs ALNS archive (HV/IGD)
python3 run_rescuenet.py <thu_muc_nhan>    # kịch bản risk map từ nhãn RescueNet thật (4 thuật toán)
python3 make_report.py                     # bảng + hình + multi_seed.csv, ablation.csv, sensitivity.csv
python3 stats_tests.py                     # Friedman, Wilcoxon + Holm, A12, Cliff's delta
python3 check_completeness.py              # kiểm số job/NaN/trùng -> tables/kiem_tra_du_lieu.md
python3 -m src.rescuenet_risk              # demo mặt nạ GIẢ -> risk map
```
Tham số ở `config.yaml` (instance, số vòng, seed, trọng số). Kết quả thô: `results/raw_*.csv`; tóm tắt: `docs/ket_qua_tom_tat.md`.

## Tái lập Pu et al. (2026) - `docs/1.pdf` (thư mục `paper_repro/`)
Bài báo: Pu, Yang, Li (2026), Aerospace Science and Technology 172:111711, DOI 10.1016/j.ast.2026.111711. File PDF (`docs/1.pdf`) không kèm trong repo vì lý do bản quyền.
Chạy bằng Python 3.11. Đặt `OMP_NUM_THREADS=1` trước khi chạy song song. Các lệnh chạy dài đều chạy tiếp được nếu bị dừng (bỏ qua phần đã xong).
```bash
py -3.11 -m pip install --user numpy scipy matplotlib scikit-learn statsmodels pytest
py -3.11 -m pytest tests/test_paper_repro.py -q
py -3.11 -m paper_repro.run_paper_repro --exp all --quick                          # thử nhanh, vài phút -> results/paper_repro/quick/
py -3.11 -m paper_repro.run_paper_repro --exp table1 table2 forecast dynamic --full # Bảng 1, 2, 7 + mô phỏng động (~20 phút)
py -3.11 -m paper_repro.forecast_seeds --seeds 1 2 3 4                             # Bảng 7 trên 4 bộ dữ liệu thêm (~30 phút)
py -3.11 -m paper_repro.run_paper_repro --exp single multi --full --workers 4      # Bảng 4, 5 (vài giờ)
py -3.11 -m paper_repro.report full                                                # sinh lại toàn bộ bảng + hình
```
CSV thô: `results/paper_repro/<quick|full>/` · bảng so sánh với bài: `tables/paper_repro/<mode>/` · hình: `figures/paper_repro/<mode>/` · báo cáo: `docs/tai_lap_pu2026.md`.

## Dữ liệu
| Bộ | Thư mục | Ghi chú |
|---|---|---|
| Solomon gốc C101, C106, C109, R101, RC101 (25/50 khách đầu) | `data/solomon_small/` | nguồn PyVRP/Instances |
| Solomon gốc 100 khách | `data/solomon_100/` | |
| Gehring–Homberger 200 khách C1_2_1, R1_2_1, RC1_2_1 | `data/homberger_200/` | chỉ loại 1 |
| RescueNet: 5 mặt nạ nhãn tập test | `data/rescuenet/` | CC BY-NC-ND 4.0; mặt nạ 10801 bị loại khỏi bảng (suy biến) |
| Instance tự sinh `SYN_*` (đợt chạy đầu) | `data/solomon_*/SYN_*` | kết quả cũ ở `results/synthetic_v1/` |

Kịch bản động (hotspot, sự kiện, đội UAV) do `src/event_generator.py` sinh, lưu ở `scenarios/`.

## Bố cục
`src/` mô hình, toán tử ALNS, mô phỏng rolling-horizon, baseline, NSGA-III · `docs/` mô hình toán, độ phức tạp, ý tưởng, ngân sách, tài liệu tham khảo · `tables/`, `figures/`, `scenarios/`, `data/`.

## Giới hạn cần nêu
- Hotspot, sự kiện và đội UAV là giả định mô hình hóa, không suy ra từ Solomon/Homberger.
- RescueNet: trọng số rủi ro theo lớp (`RISK_W`) là đề xuất, chưa có nguồn; mặt nạ bị co giãn về miền 100×100; chỉ 4 mặt nạ × 3 seed × 1 instance (RC101_050).
- Homberger 200 khách chỉ 3 instance × 5 seed; chưa có C2A, chưa dùng pymoo (NSGA-III tự viết, chưa tinh chỉnh).
- `docs/tuning_alns.md` tinh chỉnh trên một số instance `SYN_` cũ.
- Ngân sách thu nhỏ (`docs/ngan_sach_tinh_toan.md`); độ phân giải vùng cấm ≈ 2 đơn vị lưới.
