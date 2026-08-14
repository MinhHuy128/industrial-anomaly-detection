# 🛤️ CHUYÊN ĐỀ 8: LỘ TRÌNH CHẠY CHI TIẾT TỪ A-Z VÀ BÀI TOÁN TÍNH TAY
## Từ Dữ Liệu Đầu Vào $\rightarrow$ Huấn Luyện $\rightarrow$ Suy Luận $\rightarrow$ Đánh Giá AUROC/sPRO

Tài liệu này hướng dẫn toàn bộ **Vòng đời vận hành (End-to-End Lifecycle)** của mô hình ViTill-GCT V2, bao gồm:
1. **Lộ trình 7 Giai đoạn chạy thực tế.**
2. **Ví dụ tính toán Chạy tay bằng số (Dry-Run Numerical Walkthrough).**
3. **Mapping trực tiếp từng dòng code PyTorch (Line-by-Line Code Mapping).**

---

## 🗺️ 1. TỔNG QUAN LỘ TRÌNH 7 GIAI ĐOẠN (END-TO-END PIPELINE)

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 1: DATA LOADING & PREPROCESSING                                               │
│ Ảnh thô RGB -> Resize 448x448 -> CenterCrop 392x392 -> Normalize -> Tensor [B, 3, 392, 392] │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 2: FEATURE EXTRACTION (DINOv2-Register ViT-B/14)                              │
│ Tensor [B, 3, 392, 392] -> Extract 8 intermediate layers -> 8 x [B, 784, 768]          │
│ Extract CLS Token [B, 768] -> .detach() để khóa gradient                              │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 3: FUSION & BOTTLENECK & GCT TOKEN INSERTION                                  │
│ Average 8 layers -> [B, 784, 768] -> Bottleneck MLP (768->3072->768) -> [B, 784, 768]   │
│ Prepend Learned GCT Token [B, 1, 768] -> Combined Sequence [B, 785, 768]               │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 4: TRANSFORMER DECODER RECONSTRUCTION                                         │
│ Sequence [B, 785, 768] -> 8 x LinearAttention2 Blocks -> Output Sequence [B, 785, 768]│
│ Split Token 0: Decoded GCT [B, 1, 768]  |  Split Tokens 1..784: Decoded Patches [B, 784, 768]│
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼ (HUẤN LUYỆN - TRAINING)                        ▼ (SUY LUẬN - INFERENCE)
┌─────────────────────────────────────────┐     ┌─────────────────────────────────────────┐
│ GIAI ĐOẠN 5: LOSS PIPELINE & BACKPROP   │     │ GIAI ĐOẠN 6: ACTIVE DUAL-STREAM SCORING │
│ • L_rec: Top 10% Hard Patch Cosine Dist │     │ • Score_patch: Top-1% Mean Patch Error  │
│ • L_gct: 1.0 - CosSim(Proj_GCT, CLS_dino│     │ • Score_gct  : Global Cosine Distance   │
│ • L_total = L_rec + 0.5 * L_gct         │     │ • Score_final = Score_patch + 1.0*gct   │
└─────────────────────────────────────────┘     └────────────────────┬────────────────────┘
                                                                     │
                                                                     ▼
                                                ┌─────────────────────────────────────────┐
                                                │ GIAI ĐOẠN 7: SPATIAL ALIGNMENT & METRICS│
                                                │ • Map 28x28 -> Upsample 392x392         │
                                                │ • Paste vào Center (28, 28) Canvas 448x448│
                                                │ • Compute AUROC (Logical/Struct/Mean)   │
                                                │ • Compute sPRO* (Pixel localization)    │
                                                └─────────────────────────────────────────┘
```

---

## 🧮 2. VÍ DỤ CHẠY TAY BẰNG SỐ (DRY-RUN NUMERICAL WALKTHROUGH)

Để hiểu sâu bản chất toán học, hãy giả lập một mini-batch với:
- Batch size $B = 1$.
- Ảnh gồm $N = 4$ patch tokens ($2 \times 2$ ô vuông).
- Mỗi patch có độ dài đặc trưng $D = 2$ chiều.

### 📍 Giai đoạn A: Dữ liệu đặc trưng sau Bottleneck
Giả sử sau khi đi qua Bottleneck MLP, 4 patch tokens có giá trị:
$$\mathbf{X}_{\text{patch}} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \\ 0.6 & 0.8 \\ 0.8 & 0.6 \end{bmatrix} \in \mathbb{R}^{4 \times 2}$$

Và GCT Token khởi tạo học được:
$$\mathbf{t}_{\text{gct}} = \begin{bmatrix} 0.5 & 0.5 \end{bmatrix} \in \mathbb{R}^{1 \times 2}$$

Chuỗi ghép nối đi vào Decoder:
$$\mathbf{X}_{\text{in}} = \begin{bmatrix} \mathbf{t}_{\text{gct}} \\ \mathbf{X}_{\text{patch}} \end{bmatrix} = \begin{bmatrix} 0.5 & 0.5 \\ 1.0 & 0.0 \\ 0.0 & 1.0 \\ 0.6 & 0.8 \\ 0.8 & 0.6 \end{bmatrix} \in \mathbb{R}^{5 \times 2}$$

---

### 📍 Giai đoạn B: Sau khi qua Transformer Decoder
Giả sử sau 8 lớp Decoder, mô hình tái tạo lại chuỗi đầu ra $\mathbf{X}_{\text{dec}} \in \mathbb{R}^{5 \times 2}$:
$$\mathbf{X}_{\text{dec}} = \begin{bmatrix} \mathbf{t}_{\text{gct}}^{\text{out}} \\ \mathbf{D}_{\text{patch}} \end{bmatrix} = \begin{bmatrix} 0.8 & 0.2 \\ 0.95 & 0.05 \\ 0.1 & 0.9 \\ 0.2 & 0.2 \quad (\text{Patch 3 bị lỗi!}) \\ 0.75 & 0.55 \end{bmatrix}$$

---

### 📍 Giai đoạn C: Tính Điểm Lỗi Patch ($\text{Score}_{\text{patch}}$)
Công thức Cosine Distance cho từng patch $i$:
$$a(i) = 1.0 - \text{CosSim}(\mathbf{X}_{\text{patch}}[i], \mathbf{D}_{\text{patch}}[i])$$

- **Patch 1:** $\text{CosSim}([1.0, 0.0], [0.95, 0.05]) \approx 0.9986 \rightarrow a(1) = 1.0 - 0.9986 = 0.0014$ (Bình thường)
- **Patch 2:** $\text{CosSim}([0.0, 1.0], [0.1, 0.9]) \approx 0.9939 \rightarrow a(2) = 1.0 - 0.9939 = 0.0061$ (Bình thường)
- **Patch 3 (Bị lỗi):** $\text{CosSim}([0.6, 0.8], [0.2, 0.2]) = \frac{0.6 \times 0.2 + 0.8 \times 0.2}{\sqrt{0.6^2 + 0.8^2} \sqrt{0.2^2 + 0.2^2}} = \frac{0.28}{1.0 \times 0.2828} \approx 0.9899 \rightarrow a(3) = 1.0 - 0.9899 = 0.0101$
- **Patch 4:** $\text{CosSim}([0.8, 0.6], [0.75, 0.55]) \approx 0.9980 \rightarrow a(4) = 1.0 - 0.9980 = 0.0020$

Lấy `Top-1% Mean` (Patch có lỗi lớn nhất $a(3)$):
$$\text{Score}_{\text{patch}} = 0.0101$$

---

### 📍 Giai đoạn D: Tính Điểm GCT Đại Cục ($\text{Score}_{\text{GCT}}$)
Giả sử vector CLS của DINOv2 đại diện cho ảnh chuẩn "Hộp 2 kẹo + 1 bánh":
$$\mathbf{c}_{\text{dino}} = [1.0, 0.0]^T$$

Vector GCT sau khi qua Projection Head 1 lớp:
$$\mathbf{p}_{\text{gct}} = \text{LayerNorm}(\mathbf{t}_{\text{gct}}^{\text{out}}) = [0.6, 0.8]^T \quad (\text{Nhận diện sai thứ tự kẹo/bánh!})$$

Khoảng cách Cosine đại cục ($\text{Score}_{\text{GCT}}$):
$$\text{CosSim}(\mathbf{p}_{\text{gct}}, \mathbf{c}_{\text{dino}}) = \frac{1.0 \times 0.6 + 0.0 \times 0.8}{1.0 \times 1.0} = 0.600$$
$$\text{Score}_{\text{GCT}} = 1.0 - 0.600 = 0.4000 \quad (\text{BỘC PHÁT TĂNG MẠNH!})$$

---

### 📍 Giai đoạn E: Điểm Cuối Cùng Active Dual-Stream Scoring
$$\text{Score}_{\text{final}} = \text{Score}_{\text{patch}} + 1.0 \cdot \text{Score}_{\text{GCT}} = 0.0101 + 1.0 \times 0.4000 = 0.4101 \quad (\text{BÁO ĐỘNG LỖI LOGIC!})$$

---

## 💻 3. MAPPING CODE TRỰC TIẾP TỪNG BƯỚC (LINE-BY-LINE MAPPING)

| Bước Lộ Trình | File Code Cụ Thể | Đoạn Code PyTorch Thực Thi | Giải Thích Chi Tiết |
|:---|:---|:---|:---|
| **1. Data Preprocessing** | [`src/train.py`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/train.py#L300-L312) | `transforms.Compose([Transforms.Resize(448), Transforms.CenterCrop(392), Transforms.ToTensor(), Transforms.Normalize(...)])` | Co ảnh về 448x448, cắt vuông 392x392 ở giữa, chuẩn hóa ImageNet. Tensor ra: `[B, 3, 392, 392]`. |
| **2. Feature Extraction** | [`src/models/vitill_gct.py`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py#L65-L85) | `feat_list, cls_token = extract_intermediate_features(backbone, img_t, target_layers)` | Trích 8 lớp trung gian (`[B, 784, 768]`) và CLS token (`[B, 768]`). Gọi `.detach()` khóa gradient. |
| **3. Bottleneck & GCT Prepend** | [`src/models/vitill_gct.py`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py#L190-L210) | `x = self.bottleneck(sum(en_list)/len(en_list))`<br>`gct_tok = self.gct_token.expand(B, -1, -1)`<br>`x_in = torch.cat([gct_tok, x], dim=1)` | Nén đặc trưng qua Bottleneck `bMlp`. Mở rộng GCT token theo batch B và ghép vào đầu chuỗi $\rightarrow `[B, 785, 768]`. |
| **4. Linear Attention Decoder** | [`src/models/vitill_gct.py`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py#L210-L225) | `x_dec = self.decoder(x_in)`<br>`gct_out = x_dec[:, 0:1, :]`<br>`de_patches = x_dec[:, 1:, :]` | Giải nén qua 8 lớp Linear Attention $O(N)$. Tách token 0 làm Decoded GCT (`[B, 1, 768]`) và tokens 1..784 làm Decoded Patches (`[B, 784, 768]`). |
| **5. Loss & Backward (Train)** | [`src/losses/cosine_loss.py`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/losses/cosine_loss.py#L20-L40)<br>[`src/losses/gct_loss.py`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/losses/gct_loss.py#L15-L30) | `L_rec = global_cosine_hm_percent(...)`<br>`L_gct = 1.0 - F.cosine_similarity(proj_gct, cls_token.detach())`<br>`L_total = L_rec + 0.5 * L_gct`<br>`optimizer.zero_grad(); L_total.backward(); optimizer.step()` | Tính Cosine Loss tái tạo Hard-mining ($p=0.9$) + Cosine Loss GCT ($\lambda=0.5$). Tự dọn hook `h.remove()`. Thực thi lan truyền ngược update trọng số. |
| **6. Active Scoring (Inference)** | [`src/eval.py`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/eval.py#L110-L135) | `amap = compute_anomaly_map(...)`<br>`score_patch = image_score(amap)`<br>`score_final = score_patch + 1.0 * score_gct` | Tính Anomaly Map từng patch, lấy `top-1% mean` làm `Score_patch`. Cộng trực tiếp `1.0 * Score_gct` làm điểm số cuối cùng. |
| **7. Spatial Alignment & Metrics** | [`src/eval.py`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/eval.py#L60-L90)<br>[`src/eval.py`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/eval.py#L140-L185) | `a_map = F.interpolate(a_map, size=(392, 392))`<br>`canvas[28:420, 28:420] = crop_amap`<br>`auroc = compute_auroc(...)`<br>`spro = compute_spro(...)` | Upsample map về 392x392, dán vào canvas 448x448 tại `top=28, left=28` khớp GT mask. Lần lượt tính ROC-AUC cho Logical, Structural, Mean AUROC và sPRO* định vị pixel. |

---

## 🎯 4. GIẢI THÍCH CHUYÊN SÂU 2 YẾU TỐ KỸ THUẬT QUAN TRỌNG

### 1️⃣ Lý do công thức $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rec}} + \lambda \cdot \mathcal{L}_{\text{GCT}}$:
- **Tối ưu Đa Mục Tiêu (Multi-Task Learning):** $\mathcal{L}_{\text{rec}}$ đảm nhận việc ép 8 lớp Decoder tái tạo chi tiết 784 patch cục bộ để bắt vết xước (Structural). $\mathcal{L}_{\text{GCT}}$ đảm nhận việc ép GCT Token học quy luật ngữ cảnh đại cục từ DINOv2 CLS token để bắt lỗi thiếu linh kiện (Logical).
- **Cân bằng Gradient với $\lambda$ ($\lambda = 0.5$):** $\mathcal{L}_{\text{rec}}$ là trung bình trên 784 patch tokens (biên độ $0.01 \rightarrow 0.05$), còn $\mathcal{L}_{\text{GCT}}$ là khoảng cách của 1 vector đơn lẻ (biên độ $0.1 \rightarrow 0.6$). Trọng số $\lambda$ giúp cân bằng lực gradient, không để luồng GCT áp đảo luồng tái tạo patch.
- **Tại sao dùng Cosine Distance thay vì MSE/$L_2$?** Trong không gian 768 chiều, khoảng cách Euclidean ($L_2$) dễ nổ gradient do độ lớn vector. Khoảng cách Cosine ($1 - \text{CosSim}$) chỉ đo góc lệch hướng, giúp gradient ổn định và hội tụ siêu nhanh.

---

### 2️⃣ Cơ chế Layer Grouping trong GCT V2 vs Paper gốc Dinomaly:
- **Paper gốc Dinomaly:** Chia 8 lớp trung gian encoder thành 2 nhóm: `[0,1,2,3]` (nông - texture) và `[4,5,6,7]` (sâu - semantic).
- **GCT V2:** 
  - Khi đưa qua Bottleneck, mô hình cộng trung bình cả 8 lớp (`sum(feat_list)/8`) để Bottleneck có góc nhìn toàn cục tốt nhất cho GCT Token.
  - Khi tính Reconstruction Loss ($\mathcal{L}_{\text{rec}}$), mô hình vẫn giữ nguyên 2 nhóm `fuse_layer_enc` và `fuse_layer_dec` theo đúng paper gốc để so sánh Encoder-Decoder theo từng cấp độ ngữ nghĩa nông/sâu.
