# Mô hình toán: định tuyến đa UAV động có nhận biết rủi ro

Tài liệu này khớp từng công thức với mã (`src/model.py`, `src/rolling_horizon.py`). Mục 1 = T01 (ký hiệu và mô hình), mục 2 = T02 (hàm mục tiêu và ràng buộc).

## 1. Ký hiệu và mô hình (T01)

### 1.1 Tập hợp và tham số

| Ký hiệu | Ý nghĩa | Trong mã |
|---|---|---|
| $\mathcal K=\{1,\dots,K\}$ | tập UAV đồng nhất, xuất phát và kết thúc tại depot | `scn["uav"]["K"]` |
| $0$ | depot, tọa độ $p_0$ | `inst.xy[0]` |
| $\mathcal N=\{1,\dots,n\}$ | tập điểm phục vụ (khách) | `inst.n` |
| $p_i\in\mathbb R^2$ | vị trí điểm $i$ | `inst.xy[i]` |
| $q_i$ | nhu cầu (tải cần giao) | `inst.dem[i]` |
| $[e_i,\ \ell_i]$ | cửa sổ thời gian: sớm nhất / hạn (deadline) | `inst.ready[i]`, `inst.due[i]` |
| $s_i$ | thời gian phục vụ | `inst.serv[i]` |
| $r_i\ge 0$ | thời điểm điểm $i$ **xuất hiện** (biến `start` của bạn); $r_i=0$ nếu đã biết từ đầu | `scn["release"][i]` |
| $v$ | tốc độ bay | `P["speed"]` |
| $Q$ | **công suất tải** tối đa của một UAV | `inst.Q` |
| $E$, $\varepsilon$ | **pin** ban đầu và năng lượng tiêu hao mỗi đơn vị quãng đường; quãng đường tối đa $B=E/\varepsilon$ | `P["battery"]`, `P["eps"]` |
| $H$ | thời điểm kết thúc ngày bay (due của depot) | `inst.horizon` |
| $d_{ij}=\lVert p_i-p_j\rVert$ | khoảng cách Euclid | `pb.D` |

### 1.2 Trường rủi ro (risk map)

Có $M$ hotspot, hotspot $h$ có tâm $c_h(t)$ (quỹ đạo tuyến tính từng đoạn), độ rộng $\sigma_h$, cường độ $a_h(t)$:

$$r_h(x,t)=a_h(t)\,\exp\!\Big(-\frac{\lVert x-c_h(t)\rVert^2}{2\sigma_h^2}\Big),\qquad R(x,t)=1-\prod_{h}\big(1-r_h(x,t)\big)\in[0,1].$$

Hai góc nhìn của cùng một thế giới:

* **truth** $R(x,t)$: rủi ro thật, dùng để *chấm điểm* kết quả thực thi.
* **belief** $\hat R_\tau(x,t)$: điều phối viên biết gì tại thời điểm $\tau$. Hotspot chưa xuất hiện thì chưa biết; vị trí tương lai được **ngoại suy tuyến tính** theo vận tốc hiện tại, tức là không biết trước chỗ rẽ; thời điểm biến mất chưa biết. Chênh lệch truth − belief là "bất ngờ" mà thuật toán động phải xử lý.

Phơi nhiễm rủi ro của một chặng $u\to w$ khởi hành lúc $t_0$, bay thẳng với tốc độ $v$:

$$\rho(u,w,t_0)=\int_0^{d_{uw}} \hat R\big(\gamma(s),\ t_0+s/v\big)\,ds ,\qquad \gamma(s)=p_u+\tfrac{s}{d_{uw}}(p_w-p_u).$$

Trong mã, tích phân được xấp xỉ bằng $S=6$ điểm lấy mẫu dọc chặng, $t_0$ được rời rạc thành thùng rộng $\Delta=20$, và $R$ lấy trên lưới 2×2 đơn vị (độ phân giải của vùng cấm vì thế là khoảng 2 đơn vị, hotspot nên có $\sigma\ge 2$ ô lưới).

### 1.3 Trạng thái và tuyến bay

Tại thời điểm lập kế hoạch $\tau$, UAV $k$ ở vị trí $x_k$, rảnh từ $t_k\ge\tau$ (nếu đang phục vụ dở thì $x_k$ là điểm đó và $t_k$ là lúc phục vụ xong), còn quãng đường $B_k$ và tải còn lại $Q_k$. **Phần đã bay hoặc đang phục vụ được đóng băng.** Quyết định chỉ gồm tuyến còn lại $\pi_k=(i_1,\dots,i_L)$ của mỗi UAV.

Với $i_0\equiv$ vị trí hiện tại và $\mathrm{dep}_0=t_k$:

$$a_j=\mathrm{dep}_{j-1}+\frac{d_{i_{j-1}i_j}}{v},\quad \sigma_j=\max\big(a_j,\ e_{i_j},\ r_{i_j}\big),\quad \mathrm{dep}_j=\sigma_j+s_{i_j},\quad \theta_j=\max(0,\ \sigma_j-\ell_{i_j}).$$

($a_j$ giờ đến, $\sigma_j$ giờ bắt đầu phục vụ, $\theta_j$ độ trễ). UAV chờ tại chỗ nếu đến sớm. Sau điểm cuối UAV bay về depot, đến lúc $t^{\mathrm{ret}}_k$.

### 1.4 Các sự kiện động

| Sự kiện | Hiệu ứng | Mã |
|---|---|---|
| `new_target` | điểm $i$ xuất hiện tại $r_i$ | `release` |
| `risk_change` | hotspot mới xuất hiện, hoặc hotspot đổi hướng/tốc độ đột ngột | `hotspots` |
| `deadline_change` | $\ell_i\leftarrow\ell_i'$ (siết lại hoặc nới ra) | `due_hist` |
| `battery_drop` | pin còn lại của UAV $k$ giảm tỷ lệ $\phi$ | `bat *= 1-phi` |

## 2. Hàm mục tiêu và ràng buộc (T02)

Với một tập tuyến $\pi=\{\pi_k\}$, gọi $\mathcal U\subseteq\mathcal N$ là tập điểm không được phục vụ:

$$\min_{\pi}\ J(\pi)=w_d\,\underbrace{\sum_k D_k}_{\text{quãng đường}}+w_r\,\underbrace{\sum_k\sum_{\text{chặng}}\rho}_{\text{phơi nhiễm rủi ro}}+w_\ell\,\underbrace{\sum_k\sum_j\theta_j}_{\text{tổng trễ}}+w_u\,\underbrace{|\mathcal U|}_{\text{không phục vụ}}$$

Mặc định $(w_d,w_r,w_\ell,w_u)=(1,\,5,\,1,\,300)$. Số điểm trễ $\#\{j:\theta_j>0\}$ cũng được báo cáo. Chi phí "năng lượng" tỷ lệ với quãng đường ($E_k=\varepsilon D_k$), nên một số hạng quãng đường đại diện cho cả hai.

**Nhìn đa mục tiêu** (dùng cho NSGA-III, Hypervolume, IGD):
$$f(\pi)=\Big(\sum_k D_k,\ \sum \rho,\ \sum\theta_j+\tfrac{w_u}{w_\ell}|\mathcal U|\Big).$$

### Ràng buộc cứng (mỗi tuyến $\pi_k$)

| # | Ràng buộc | Biểu thức | Mã |
|---|---|---|---|
| C1 | Pin (gồm cả chặng về depot) | $D_k\le B_k$ | `dist > pb.bat[k]` |
| C2 | Tải trọng | $\sum_{i\in\pi_k}q_i\le Q_k$ | `load > pb.cap[k]` |
| C3 | Vùng cấm | mọi chặng không có điểm nào với $\hat R>\theta_{\mathrm{ban}}$ (không tính trong bán kính $r_0=8$ quanh depot) | `BAN[kb][u][w]` |
| C4 | Kết thúc ngày bay | $t^{\mathrm{ret}}_k\le H$ | `tret > pb.Tend` |
| C5 | Mỗi điểm tối đa một lần | các $\pi_k$ rời nhau | `check_plan` |

**Ràng buộc mềm**: cửa sổ thời gian. Trễ qua hạn $\ell_i$ bị phạt $w_\ell\theta_j$; điểm quá trễ hoặc không khả thi bị bỏ và chịu $w_u$. Vì vậy "deadline đổi" biến thành thay đổi trong $\theta_j$, không làm bài toán vô nghiệm.

**Phản xạ an toàn khi thực thi** (chung cho mọi thuật toán): trước khi bay tới điểm kế tiếp, nếu pin không đủ để tới đó rồi về depot thì bỏ điểm đó; nếu không còn đủ pin để về depot thì UAV coi như mất (`n_lost`). Đây là tầng phản xạ tối thiểu, độc lập với bộ điều phối.

### Chỉ số thực thi (dùng để so sánh thuật toán)

Sau khi UAV bay thật trên risk map **truth**, các đại lượng $D,\ \Phi,\ T,\ |\mathcal U|$ và $J$ được tính lại bằng đúng công thức trên (`Sim.metrics()`), cộng thêm `ban_dist` (quãng đường thực sự bay trong vùng $R>\theta_{\mathrm{ban}}$, ngoài bán kính an toàn của depot), `n_lost`, `replans`, `reassign` (số lần một điểm đã gán bị đổi UAV hoặc bị bỏ sau mỗi lần lập lại kế hoạch) và `plan_time`.

## 3. Giả định và giới hạn (cần nêu khi viết báo cáo)

1. UAV đồng nhất, một chuyến bay trong ngày (không sạc), bay thẳng trên mặt phẳng 2D, không va chạm giữa các UAV (đúng như giả định đơn giản hóa của Pu et al.).
2. Hotspot, sự kiện và đội UAV là giả định mô hình hóa, không suy ra từ Solomon.
3. Độ phân giải vùng cấm khoảng 2 đơn vị (lưới), phơi nhiễm là xấp xỉ lấy mẫu; kiểm tra bằng `tests/test_operators.py::test_table_risk_matches_continuous_sampling`.
4. Khi một UAV đã chờ tại điểm phục vụ, nó được coi là cam kết phục vụ điểm đó (không điều đi nơi khác).
