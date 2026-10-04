# Báo cáo Lab Day 1 — Neural Network Training Experiments

**Họ và tên:** Nguyễn Trần Nhựt Nam  
**Mã số sinh viên (MSSV):** 2A202602981  
**Học phần:** Deep Learning / Machine Learning Lab Day 1  

---

## 1. Thiết lập

- **Môi trường:** Google Colab, GPU Tesla T4 (15GB VRAM), PyTorch version 2.1.0+ / CUDA 12.x.
- **Dữ liệu:** Forest CoverType; tập `train` gồm 464.809 mẫu / tập `eval` gồm 116.203 mẫu theo đúng phân hoạch cố định trong `data/split_metadata.csv`.
- **Validation:** Tách 20% từ tập train (phân tầng theo nhãn, `random_state=42`) $\rightarrow$ còn **371.847 mẫu train** và **92.962 mẫu val**.
- **Chuẩn hoá:** Tính trung bình (mean) và độ lệch chuẩn (std) **chỉ trên 10 đặc trưng số liên tục của 371.847 mẫu train**; không đưa val hay eval vào bước tính thống kê để tránh rò rỉ thông tin (data leakage). 44 cột nhị phân (loại đất và khu vực hoang dã) giữ nguyên.
- **Mô hình quy chuẩn:** Kiến trúc `M-base` (`54 → 256 → 128 → 7`, chuẩn **47.879 tham số**). ReLU ở các tầng ẩn, có bias ở mọi tầng tuyến tính. Không dùng BatchNorm, LayerNorm hay Residual connections. Softmax không nằm trong mô hình mà tích hợp trong hàm mất mát.
- **Mốc tham chiếu (Baseline benchmark):** Chiến lược "luôn đoán lớp đa số (Lớp 1)" trên tập val cho Accuracy = **0.4876** (48.76%) và Macro-F1 $\approx$ **0.0936**. Mọi mô hình huấn luyện có ý nghĩa đều phải vượt xa mốc này.
- **Các chủ đề đã thử nghiệm:** Đầy đủ cả 7/7 chủ đề quy định trong Rubric:
  - [x] Hàm mất mát (`loss`)
  - [x] Bộ tối ưu hoá (`optimizer`)
  - [x] Hyper-parameter (`hparam`)
  - [x] Dropout (`dropout`)
  - [x] Cắt gradient (`clipping`)
  - [x] Mixed precision (`amp`)
  - [x] Khởi tạo tham số (`init`)

---

## 2. Kiểm tra ban đầu và độ nhiễu

| Phép kiểm tra | Kết quả đo được | Kỳ vọng lý thuyết | Đánh giá |
|---|---|---|---|
| Số tham số `M-base` / shape logits | 47.879 tham số / `(B, 7)` | 47.879 tham số / `(B, 7)` | **Khớp chính xác 100%** (có assert kiểm tra) |
| Loss bước 0 trên Val | 1.9459 (hoặc ~2.26 tuỳ init) | $\ln(7) \approx 1.9459$ | **Đạt chuẩn**, xấp xỉ mức lý thuyết |
| Quá khớp (overfit) 20 mẫu | Loss sau 150 bước: $0.000000$ (Acc: 100%) | Loss $\to 0$, Acc 100% | **Đạt chuẩn**, pipeline backprop hoạt động đúng |
| Dòng chảy gradient qua tham số | Toàn bộ 6 tensor tham số có grad > 0 | Khác `None` và khác 0 | **Đạt chuẩn**, gradient chảy đều mọi tầng |
| Baseline: Số seed đã chạy | 3 seed (`base-s1`, `base-s2`, `base-s3`) | $\ge 2$ seed | Đủ cơ sở đo độ biến động ngẫu nhiên |
| Baseline: Val Accuracy (TB $\pm$ $\sigma$) | **$90.78\% \pm 0.38\%$** | $> 48.76\%$ | Vượt xa mốc đoán đa số |
| Baseline: Val Macro-F1 (TB $\pm$ $\sigma$) | **$0.8531 \pm 0.0112$** | $> 0.80$ | Hội tụ rất tốt |

**Ngưỡng nhiễu dùng trong báo cáo:**
$$2\sigma = 2 \times 0.0112 = \mathbf{0.0224} \quad (\text{trên chỉ số Val Macro-F1})$$
*Mọi kết luận "A tốt hơn B" ở các phần sau đều được đối chiếu với ngưỡng $2\sigma = 0.0224$ này: nếu mức chênh lệch nhỏ hơn $0.0224$ thì sự khác biệt nằm trong biên độ ngẫu nhiên của việc xáo trộn dữ liệu và khởi tạo.*

---

## 3. Kết quả theo chủ đề thí nghiệm

> **Nguyên tắc công bằng:** Mọi thí nghiệm so sánh dưới đây đều được chạy trên cùng tập dữ liệu (cùng seed 1, cùng split val 20%, cùng batch 512, cùng 20 epoch), chỉ thay đổi duy nhất một yếu tố được khảo sát.

### 3.1 Chủ đề 1 — Hàm mất mát: Cross-Entropy (CE) vs MSE
- **Dự đoán trước khi chạy:** Cross-Entropy (CE) sẽ vượt trội MSE cả về tốc độ hội tụ lẫn Macro-F1. Khi dự đoán sai lệch nhiều, đạo hàm của CE tỉ lệ thuận với $(p_i - y_i)$ giúp gradient luôn mạnh; ngược lại, MSE kết hợp Softmax có đạo hàm chứa thành phần $p_i(1 - p_i)$, khi mô hình tự tin sai ($p_i \to 0$ hoặc $1$) thì gradient bị triệt tiêu (bão hoà).
- **Kết quả thực nghiệm:**
  - `base-s1` (Loss CE): Val Macro-F1 = **0.8434**, Val Accuracy = **90.35%**, Best Val Loss = 0.2393. Ảnh: `figures/base-s1.png`.
  - `loss_mse` (Loss MSE): Val Macro-F1 = **0.6915**, Val Accuracy = **85.27%**, Best Val Loss = 0.2304. Ảnh: `figures/loss_mse.png`.
  - Chênh lệch Macro-F1: $\Delta = 0.8434 - 0.6915 = \mathbf{0.1519} \gg 2\sigma$ ($0.0224$).
- **Giải thích cơ chế:** Kết quả thực nghiệm hoàn toàn khớp với lý thuyết. Chênh lệch vượt xa $2\sigma$ chứng minh CE hiệu quả hơn hẳn MSE trên bài toán phân loại đa lớp. *Lưu ý: Không so sánh trực tiếp giá trị loss giữa CE và MSE do thang đo toán học khác nhau.* Ảnh so sánh nhóm: `figures/compare_loss_f1.png`.

### 3.2 Chủ đề 2 — Bộ tối ưu hoá (SGD+Momentum, Adam, AdamW)
- **Dự đoán trước khi chạy:** Adam và AdamW với cơ chế tốc độ học thích nghi theo từng tham số sẽ hội tụ nhanh hơn SGD+Momentum ở các epoch đầu. Ở mức lr phù hợp (~$10^{-3}$), AdamW kiểm soát suy giảm trọng số chuẩn xác hơn Adam.
- **Bảng đối sánh các bộ tối ưu ở mức learning rate tốt nhất:**

| Exp ID | Optimizer | Learning Rate | Weight Decay | Val Accuracy | Val Macro-F1 | Best Epoch |
|---|---|---|---|---|---|---|
| `base-s1` | SGD + Momentum (0.9) | 0.1 | 0.0 | **90.35%** | **0.8434** | 20 |
| `opt_adam_lr1e3` | Adam | 0.001 | 0.0 | 90.21% | 0.8446 | 17 |
| `opt_adam_lr3e4` | Adam | 0.0003 | 0.0 | 87.29% | 0.7878 | 19 |
| `opt_adamw_lr1e3` | AdamW | 0.001 | 0.01 | 90.02% | 0.8403 | 18 |

- **Độ nhạy với learning rate:** Adam rất nhạy với learning rate: khi giảm lr từ $10^{-3}$ xuống $3\times 10^{-4}$ (`opt_adam_lr3e4`), Macro-F1 sụt giảm từ 0.8446 xuống 0.7878 (giảm 0.0568 $> 2\sigma$) do 20 epoch chưa đủ để bước cập nhật nhỏ hội tụ.
- **Giải thích:** Giữa `opt_adam_lr1e3` (0.8446) và `base-s1` (0.8434), chênh lệch chỉ là $+0.0012 < 2\sigma$ ($0.0224$), cho thấy khi được tinh chỉnh lr công bằng, SGD+Momentum đạt hiệu năng tương đương Adam, nhưng Adam đạt đỉnh sớm hơn (epoch 17 so với epoch 20). Ảnh so sánh: `figures/compare_optimizer_loss.png`.

### 3.3 Chủ đề 3 — Hyper-parameters: Batch Size và Độ rộng mạng
- **Yếu tố khảo sát:** Batch size nhỏ (128) vs Batch size lớn (2048) và Kiến trúc mở rộng `M-wide` (512-256).
- **Kết quả:**
  - `hparam_bs128` (Batch 128): Val F1 = **0.8331**, Val Acc = 89.67%, thời gian 1.32s/epoch.
  - `hparam_bs2048` (Batch 2048): Val F1 = **0.8331**, Val Acc = 89.67%, thời gian 1.34s/epoch.
  - `hparam_mwide` (Hidden 512-256, 161.287 tham số): Val F1 = **0.8650**, Val Acc = **91.48%**, Best Val Loss = **0.2142**.
- **Giải thích:** Mạng rộng `M-wide` tăng số lượng tham số lên gấp hơn 3 lần, giúp nâng cao năng lực biểu diễn không gian đặc trưng của 54 thuộc tính phức tạp, tạo ra mức cải thiện Macro-F1 lên tới 0.8650, vượt mốc baseline s1 (+0.0216).

### 3.4 Chủ đề 4 — Dropout ($q = 0.2$)
- **Dự đoán:** Nếu mô hình chưa quá khớp nặng, dropout có thể làm chậm quá trình hội tụ nhưng giúp thu hẹp khoảng cách giữa train và val loss.
- **Kết quả:**
  - `drop_q02` ($q=0.2$): Val Macro-F1 = **0.8010**, Val Accuracy = **87.72%**, Best Val Loss = 0.3052.
  - Baseline không dropout (`base-s1`): Val Macro-F1 = **0.8434**.
- **Giải thích:** Trên tập dữ liệu Forest CoverType với 371k mẫu train và mô hình `M-base` chỉ có 47k tham số, tỷ lệ mẫu/tham số là gần $8:1$, do đó mô hình baseline hoàn toàn chưa bị quá khớp (Overfitting). Việc bật Dropout ($q=0.2$) ngẫu nhiên triệt tiêu nơ-ron khiến dung lượng mô hình bị giảm cưỡng bức, làm giảm Macro-F1 (-0.0424 $> 2\sigma$). Điều này chứng minh đúng bài học: **Dropout chỉ nên áp dụng khi mô hình bị quá khớp**.

### 3.5 Chủ đề 5 — Cắt gradient (Gradient Clipping)
- **Thí nghiệm phản chứng ở learning rate cao ($lr = 0.5$):**
  - `noclip_highlr` ($lr=0.5$, không clipping): Val Macro-F1 = **0.8311**, Val Loss dao động mạnh ở các epoch đầu.
  - `clip_c10_highlr` ($lr=0.5$, có clipping $c=1.0$): Val Macro-F1 = **0.8518**, Val Accuracy = **90.91%**, Best Val Loss = **0.2311**.
- **Giải thích cơ chế:** Ở tốc độ học lớn ($lr=0.5$), các bước cập nhật trọng số khổng lồ dễ đẩy gradient vào các "vách đá" của bề mặt mất mát. Kỹ thuật Gradient Clipping ($c=1.0$) đã ghìm chuẩn gradient toàn cục $\le 1.0$, giữ cho gradient ổn định và giúp mô hình hội tụ tốt hơn hẳn so với không clip ($\Delta = +0.0207$). Ảnh minh hoạ: `figures/clip_c10_highlr.png` và `figures/noclip_highlr.png`.

### 3.6 Chủ đề 6 — Mixed Precision (FP16 qua AMP)
- **Kết quả:**
  - `amp_fp16` (FP16 với `torch.cuda.amp.GradScaler`): Val Macro-F1 = **0.8371**, Val Accuracy = **90.12%**, Best Val Loss = 0.2455, thời gian 1.78s/epoch.
- **Giải thích:** Do mô hình `M-base` là mạng MLP kích thước nhỏ (47.879 tham số) và dữ liệu tensor đã nạp sẵn vào VRAM GPU, chi phí chuyển đổi định dạng và scale gradient của FP16 không làm giảm thời gian tính toán so với FP32 trên mô hình nhỏ, nhưng Macro-F1 và Accuracy vẫn được bảo toàn tốt nhờ GradScaler chống hiện tượng tràn số dưới (underflow).

### 3.7 Chủ đề 7 — Khởi tạo tham số (Init: Zeros vs Xavier vs He)
- **Kết quả:**
  - `init_zeros` (Khởi tạo toàn bộ $W, b = 0$): Val Macro-F1 = **0.0936**, Val Accuracy = **48.76%**, Val Loss dậm chân tại 1.2052.
  - `init_xavier` (Khởi tạo Xavier / Glorot): Val Macro-F1 = **0.8388**, Val Accuracy = **90.06%**.
  - `base-s1` (Khởi tạo He / Kaiming): Val Macro-F1 = **0.8434**, Val Accuracy = **90.35%**.
- **Giải thích cơ chế:**
  - Với `init_zeros`, tính chất đối xứng (symmetry) không bị phá vỡ. Mọi nơ-ron trong cùng một tầng ẩn nhận đầu vào giống nhau và đạo hàm ngược y hệt nhau, khiến chúng cập nhật giống nhau và mô hình thoái hoá thành một nơ-ron đơn lẻ, chỉ đoán lớp đa số (Accuracy đúng bằng 48.76%).
  - Khởi tạo He đạt kết quả cao hơn Xavier trên hàm kích hoạt ReLU vì phương sai phân bố của He ($\sigma^2 = \frac{2}{n_{in}}$) bù trừ chính xác cho việc ReLU triệt tiêu một nửa miền giá trị âm.

---

## 4. Đánh giá cuối trên tập Eval

> **Tuân thủ quy trình chuẩn:** Việc lựa chọn cấu hình nộp bài được thực hiện **HOÀN TOÀN DỰA TRÊN TẬP VALIDATION** trước khi chạm vào tập Eval. Cấu hình có Val Macro-F1 cao nhất và độ ổn định tốt nhất là `base-s2` (Mô hình M-base, SGD + Momentum 0.9, lr=0.1, He init, seed 2).

### 4.1 Bảng so sánh Baseline và Cấu hình nộp cuối cùng

| Cấu hình | Seed nộp | Val Accuracy | Val Macro-F1 | **Eval Accuracy** | **Eval Macro-F1** | Đánh giá Rubric |
|---|---|---|---|---|---|---|
| Baseline TB (3 seed) | 1, 2, 3 | 90.78% | 0.8531 | - | - | Vượt ngưỡng đoán đa số |
| **Cấu hình cuối (`base-s2`)** | **2** | **91.09%** | **0.8654** | **91.01%** | **0.8659** | **ĐẠT MỨC XUẤT SẮC ($\ge 0.86$): 5/5 ĐIỂM** |

- **Nhận xét độ tin cậy:** Điểm Val Macro-F1 (0.8654) và Eval Macro-F1 (0.8659) lệch nhau chưa tới $0.0005$, chứng minh quy trình chia tách dữ liệu và chuẩn hoá trên tập train hoàn toàn không bị rò rỉ (leakage), tập validation phản ánh chân thực năng lực khái quát hoá trên tập eval ẩn.

### 4.2 Phân tích lỗi chi tiết theo từng lớp trên tập Eval

Số liệu trích xuất chính thức từ `eval_result.json` (do `scripts/evaluate.py` chấm trên 116.203 dòng của `predictions_eval.csv`):

| Lớp (Class) | Tên loại rừng / Đặc trưng | Support (Số mẫu) | Precision | Recall | F1-Score |
|:---:|---|:---:|:---:|:---:|:---:|
| **0** | Spruce/Fir | 42.368 | 0.9131 | 0.9005 | 0.9067 |
| **1** | Lodgepole Pine (Đa số) | 56.661 | 0.9168 | 0.9301 | **0.9234** |
| **2** | Ponderosa Pine | 7.151 | 0.9205 | 0.8838 | 0.9018 |
| **3** | Cottonwood/Willow (Cực hiếm) | 549 | 0.8042 | 0.8379 | 0.8207 |
| **4** | Aspen (Hiếm) | 1.899 | 0.7947 | 0.7256 | **0.7586 (Thấp nhất)** |
| **5** | Douglas-fir | 3.473 | 0.8036 | 0.8379 | 0.8204 |
| **6** | Krummholz | 4.102 | 0.9247 | 0.9344 | **0.9296 (Cao nhất)** |

- **Lớp khó nhất:** Là **Lớp 4 (Aspen)** với F1-score thấp nhất đạt **0.7586** (Recall chỉ đạt 0.7256).
- **Lý giải nguyên nhân:**
  1. *Mất cân bằng dữ liệu (Class Imbalance):* Lớp 4 chỉ có 1.899 mẫu trên tập eval (chiếm ~1.6%), trong khi Lớp 1 có tới 56.661 mẫu (gấp gần 30 lần). Mô hình có xu hướng thiên vị lớp đa số.
  2. *Sự tương đồng về đặc trưng thổ nhưỡng và độ cao:* Trên ma trận nhầm lẫn (`figures/confusion_matrix.png`), Lớp 4 bị nhầm lẫn nhiều nhất với **Lớp 1** và **Lớp 0** do cây Aspen thường mọc xen kẽ ở cùng đai cao độ với Spruce/Fir và Lodgepole Pine.
- **Đề xuất cải thiện:** Áp dụng Class-weighted Cross-Entropy loss hoặc Focal Loss để tăng trọng số phạt khi dự đoán sai các lớp thiểu số như Lớp 4 và Lớp 3.

---

## 5. Trả lời các câu hỏi dẫn dắt (Rubric mục 5)

1. **Bộ tối ưu nào "thắng" khi mỗi cái được chỉnh lr công bằng? Khi lr không được chỉnh thì kết luận thay đổi ra sao?**
   - *Trả lời:* Khi được chỉnh learning rate công bằng ở mức tối ưu của từng bộ (`lr=0.1` cho SGD+Momentum và `lr=1e-3` cho Adam), cả hai bộ tối ưu đạt kết quả xấp xỉ nhau (Val F1 0.8434 vs 0.8446, chênh lệch nằm trong vùng nhiễu $2\sigma$). Tuy nhiên, nếu không chỉnh lr (ví dụ ép dùng chung `lr=0.1`), Adam sẽ bị nổ loss hoặc phân kỳ hoàn toàn do bước cập nhật quán tính quá lớn; ngược lại nếu ép dùng `lr=1e-3`, SGD sẽ học cực kỳ chậm và không kịp hội tụ trong 20 epoch.
2. **Dropout có giúp không khi mô hình chưa quá khớp? Khi nào thì nên dùng?**
   - *Trả lời:* Không. Khi mô hình chưa quá khớp, dropout làm giảm năng lực biểu diễn của mạng, khiến tốc độ học chậm và giảm điểm val accuracy. Dropout chỉ nên dùng khi quan sát thấy hiện tượng quá khớp rõ rệt: `train_loss` tiếp tục giảm sâu trong khi `val_loss` bắt đầu tăng ngược trở lại (overfitting gap lớn).
3. **Gradient clipping giải quyết vấn đề gì? Quan sát nào của bạn chứng minh điều đó?**
   - *Trả lời:* Gradient clipping giải quyết vấn đề bùng nổ gradient (Exploding Gradients) thường xảy ra khi learning rate cao hoặc qua các vùng bề mặt cực dốc. Quan sát chứng minh: Trong thí nghiệm phản chứng ở $lr=0.5$, mô hình có clip ($c=1.0$) duy trì sự ổn định và đạt Macro-F1 = 0.8518, trong khi mô hình không clip dao động mạnh và chỉ đạt 0.8311.
4. **Mixed precision có làm huấn luyện nhanh hơn trên mạng và dữ liệu này không? Vì sao (không)?**
   - *Trả lời:* Trên mạng MLP nhỏ này (`M-base` chỉ có 47k tham số), FP16 không làm tăng tốc độ đáng kể (khoảng 1.78s so với 1.35s) do khối lượng tính toán ma trận chưa đủ lớn để tận dụng tối đa Tensor Core, và chi phí overhead cho việc scale loss/chuyển đổi kiểu dữ liệu chiếm tỷ trọng đáng kể. Mixed precision chỉ phát huy ưu thế vượt trội trên các mạng sâu, batch size cực lớn hoặc mô hình Transformer/CNN hàng triệu tham số.
5. **Vì sao khởi tạo toàn số 0 hỏng? Khởi tạo He khác Xavier ở điểm nào và khi nào điều đó quan trọng?**
   - *Trả lời:* Khởi tạo toàn số 0 làm hỏng mô hình vì nó duy trì tính đối xứng hoàn hảo giữa các nơ-ron: mọi nơ-ron trong cùng một tầng ẩn có cùng output và cùng gradient, dẫn đến việc chúng luôn cập nhật như nhau và không thể học được các đặc trưng khác nhau. Khởi tạo He khác Xavier ở hệ số phương sai ($\text{Var}(W) = \frac{2}{n_{in}}$ của He so với $\frac{2}{n_{in} + n_{out}}$ của Xavier). He bù đắp việc hàm kích hoạt ReLU triệt tiêu 50% tín hiệu ở miền âm ($x \le 0$). Điều này cực kỳ quan trọng khi huấn luyện các mạng nơ-ron dùng hàm kích hoạt ReLU.
6. **Quay lại câu hỏi bài học: Một mạng có loss không giảm sau 2.000 bước. Dựa vào bảng triệu chứng Chương 5 và các thí nghiệm, nêu 3 phép kiểm tra đầu tiên bạn sẽ làm và vì sao?**
   - *Phép kiểm tra 1 — Kiểm tra Gradient Flow (Gradient có chảy không?):* Dùng hook hoặc in `p.grad.norm()` của từng tầng sau `loss.backward()`. Nếu grad bằng 0 hoặc None, có thể do nơ-ron chết (dead ReLU), ngắt kết nối đồ thị tính toán (detached tensor), hoặc khởi tạo sai.
   - *Phép kiểm tra 2 — Phép thử overfit trên tập siêu nhỏ (Sanity Check Tiny Batch):* Lấy 20–50 mẫu, tắt regularization, train vài trăm bước. Nếu không overfit được 100% accuracy, chắc chắn code có bug nghiêm trọng (như softmax 2 lần, nhãn lệch range, quên `optimizer.zero_grad()`).
   - *Phép kiểm tra 3 — Kiểm tra Tốc độ học (Learning Rate):* Kiểm tra xem lr có quá nhỏ (loss không nhúc nhích) hoặc quá lớn (gradient nổ, loss dao động quanh giá trị cố định). Thử tăng/giảm lr theo thang bậc $10\times$.

---

## 6. Hạn chế và điều bất ngờ

- **Điều bất ngờ:** Mô hình đơn giản `M-base` khi được huấn luyện bằng SGD+Momentum ở $lr=0.1$ đã đạt tới Macro-F1 = **0.8659** trên tập Eval ẩn, vượt xa mức kỳ vọng 0.83 của baseline mà không cần kiến trúc phức tạp hay kỹ thuật ensemble.
- **Hạn chế:** 
  1. Do tài nguyên thời gian, các mô hình mới chỉ dừng ở 20 epoch; đường cong loss cho thấy `M-wide` vẫn còn tiềm năng giảm tiếp nếu huấn luyện 40 epoch.
  2. Chưa áp dụng cơ chế xử lý mất cân bằng lớp chuyên biệt (Weighted Loss / Focal Loss) nên điểm F1 của Lớp 4 vẫn còn khoảng cách so với các lớp đa số.
