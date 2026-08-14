# 🏗️ CHUYÊN ĐỀ 2: KIẾN TRÚC MÔ HÌNH VITILL-GCT V2 CHI TIẾT (ARCHITECTURE DEEP-DIVE)

---

## 1. SƠ ĐỒ KHỐI TỔNG THỂ VÀ LUỒNG MA TRẬN DỮ LIỆU (MATRIX DATAFLOW)

Dưới đây là sơ đồ luồng dữ liệu biến đổi ma trận Tensor từ ảnh đầu vào $448 \times 448 \times 3$ đến khi tính Anomaly Score:

```text
[Input Image: 3 x 448 x 448]
        │
        ▼  Resize & CenterCrop (392 x 392)
[Cropped Image: 3 x 392 x 392]
        │
        ▼  DINOv2-Register Backbone ViT-B/14 (FROZEN, 86M Params)
[Intermediate Layers: 8 layers x (B, 784, 768)] + [CLS Token: B x 768 (DETACHED)]
        │
        ▼  Layer Fusion & Bottleneck MLP: bMlp(768 -> 3072 -> 768)
[Fused Patch Features: B x 784 x 768]
        │
        ▼  Prepend Learned GCT Token: t_gct (B x 1 x 768)
[Combined Input Sequence: B x 785 x 768]
        │
        ▼  8-Layer Linear Attention Transformer Decoder
[Decoded Patch Features: B x 784 x 768]  +  [Decoded GCT Token: B x 1 x 768]
        │                                             │
        ▼                                             ▼  Projection Head (Linear + LN)
[Per-patch Cosine Distance]                   [Proj GCT Vector: B x 768]
        │                                             │
        ▼                                             ▼  Cosine Distance vs DINO CLS
[Patch Score: Top-1% Mean]                    [GCT Score: Global Cosine Distance]
        │                                             │
        └──────────────────────┬──────────────────────┘
                               ▼
        [Final Score = Patch Score + 1.0 * GCT Score]
```

---

## 2. PHÂN TÍCH 5 THÀNH PHẦN KIẾN TRÚC CỐT LÕI (5 CORE SUBSYSTEMS)

### 1️⃣ Backbone Encoder: DINOv2-Register ViT-B/14
- **Trực giác:** Bộ trích xuất đặc trưng mắt thần siêu việt được Facebook/Meta huấn luyện tự giám sát (Self-Supervised Learning) trên hàng trăm triệu bức ảnh.
- **Tại sao lại dùng bản `-Register` (`dinov2_vitb14_reg`)?**
  - Mẫu DINOv2 thông thường có nhược điểm bị nhiễu các điểm cực trị vô nghĩa (Artifact Spikes) ở các vùng nền tĩnh. 
  - Bản `-Register` thêm 4 `Register Tokens` đóng vai trò là "hố thu rác nhiễu", giúp các patch tokens còn lại cực kỳ sạch sẽ và mượt mà.
- **Kích thước Ma trận:**
  - Ảnh vào: $3 \times 392 \times 392$. Với kích thước patch $14 \times 14$, ta có:
    $$N = \frac{392}{14} \times \frac{392}{14} = 28 \times 28 = 784 \quad \text{Patch Tokens}$$
  - Trích xuất 8 lớp trung gian (Intermediate Layers `[2, 3, 4, 5, 6, 7, 8, 9]`), mỗi lớp cho ra Tensor shape $[B, 784, 768]$.
  - Vector CLS token: $[B, 768]$. **BẮT BUỘC KHÓA GRADIENT bằng `.detach()`** để không làm hỏng trọng số backbone!

---

### 2️⃣ Bottleneck MLP (`bMlp`)
- **Trực giác:** Phễu nén đặc trưng. Nhận đặc trưng từ 8 lớp của encoder, nén lại và biến đổi phi tuyến trước khi đưa qua Decoder.
- **Cấu trúc:**
  - `Linear(768, 3072)` $\rightarrow$ `GELU` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Linear(3072, 768)`
- **Tác dụng:** Giảm hiện tượng overfitting và tạo ra khoảng không gian biểu diễn chung (Shared Latent Space) cho các patch tokens.

---

### 3️⃣ GCT Token & Projection Head
- **Trực giác:** Token học được được ghép vào đầu chuỗi patch tokens. Nối với một đầu chiếu (Projection Head) 1 lớp để so sánh với vector đại cục của DINOv2.
- **Kích thước:**
  - Token khởi tạo: $\mathbf{t}_{\text{gct}} \in \mathbb{R}^{1 \times 1 \times 768}$ (Parameter học được).
  - Chuỗi ghép vào Decoder: $[B, 1 + 784, 768] = [B, 785, 768]$.
  - Projection Head: `Linear(768, 768)` + `LayerNorm(768)`.

---

### 4️⃣ 8-Layer Linear Attention Transformer Decoder
- **Trực giác:** Transformer Decoder dùng cơ chế Attention tuyến tính để khôi phục ảnh từ đặc trưng nén.
- **Tại sao lại dùng `Linear Attention` thay vì `Standard Softmax Attention`?**
  - Softmax Attention tiêu tốn bộ nhớ và độ phức tạp tính toán $O(N^2)$ với $N=784 \rightarrow 784^2 = 614,656$ phép tính.
  - Linear Attention (dùng hàm kích hoạt ELU+1) giảm độ phức tạp xuống $O(N)$, giúp tốc độ inference siêu nhanh **~87ms/ảnh (11.4 FPS)** trên GPU!

---

### 5️⃣ Loss Pipeline & Dual-Stream Active Scoring
- **Loss Tái Tạo (Reconstruction Loss):** Khoảng cách Cosine theo từng patch giữa Encoder và Decoder với cơ chế Hard-mining ($p=0.9$ chọn 10% patch khó nhất):
  $$\mathcal{L}_{\text{rec}} = \text{Mean of Top 10% hardest patch cosine distances}$$
- **Loss GCT:** Khoảng cách Cosine giữa Proj GCT Token và DINO CLS Token:
  $$\mathcal{L}_{\text{GCT}} = 1.0 - \text{CosineSimilarity}(\mathbf{p}_{\text{gct}}, \mathbf{c}_{\text{dino}}.\text{detach}())$$
- **Tổng Loss Huấn Luyện:**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rec}} + 0.5 \cdot \mathcal{L}_{\text{GCT}}$$
- **Active Dual-Stream Scoring (Inference):**
  $$\text{Score}_{\text{final}} = \text{Score}_{\text{patch}} (\text{Top-1\% mean}) + 1.0 \cdot \text{Score}_{\text{GCT}}$$

---

## 3. MAPPING ĐẾN FILE CODE PROJECT

- Lớp Bottleneck & Decoder: [`src/models/vitill_gct.py`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py#L90-L160)
- Toàn bộ Mô hình ViTillGCT: [`src/models/vitill_gct.py`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py#L165-L240)
- Hàm Loss Cosine Tái tạo: [`src/losses/cosine_loss.py`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/losses/cosine_loss.py#L10-L45)
- Hàm Loss GCT: [`src/losses/gct_loss.py`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/losses/gct_loss.py#L10-L35)
