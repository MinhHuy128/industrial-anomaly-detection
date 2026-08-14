# 🎤 SCRIPT THUYẾT TRÌNH 20 PHÚT — VITILL-GCT V2
## Đi thẳng vào Model, Architecture, Code, Kết quả

---

## 🗺️ PHÂN BỐ THỜI GIAN 20 PHÚT

| Phần | Nội Dung | Thời Gian |
|:---:|:---|:---:|
| **1** | Kiến trúc tổng quan Pipeline (show sơ đồ) | 4 phút |
| **2** | Chi tiết từng khối: Input/Output/Dimension | 5 phút |
| **3** | GCT Token + Loss Pipeline (phần mình làm) | 5 phút |
| **4** | Kết quả thực nghiệm + so sánh Baseline | 3 phút |
| **5** | Bug đã fix + Q&A buffer | 3 phút |

---

## ▶️ PHẦN 1 (Phút 0:00 – 4:00): KIẾN TRÚC TỔNG QUAN (SHOW SƠ ĐỒ)

### 🖥️ SHOW GÌ:
> Mở và chiếu **bức ảnh sơ đồ Full Pipeline** (file ảnh sơ đồ 7 giai đoạn).

### 🗣️ NÓI GÌ:

> *"Em xin báo cáo trực tiếp vào phần mô hình luôn ạ.*
>
> *Đây là toàn bộ Pipeline mô hình của em — ViTill-GCT V2. Gồm 7 giai đoạn, chia thành 2 nhánh chính:*
> - ***Nhánh trái (Training):** Input ảnh $\rightarrow$ Backbone trích đặc trưng $\rightarrow$ Bottleneck + GCT Token $\rightarrow$ Decoder $\rightarrow$ Tính Loss $\rightarrow$ Backward cập nhật trọng số.*
> - ***Nhánh phải (Inference):** Chạy qua mô hình đã train $\rightarrow$ Tính điểm bất thường theo luồng đôi $\rightarrow$ Căn chỉnh tọa độ $\rightarrow$ Tính AUROC và sPRO.*
>
> *Em sẽ đi qua từng giai đoạn chi tiết ạ."*

---

## ▶️ PHẦN 2 (Phút 4:00 – 9:00): CHI TIẾT TỪNG KHỐI — INPUT / OUTPUT / DIMENSION

### 🖥️ SHOW GÌ:
> Mở VS Code, mở file [`src/models/vitill_gct.py`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py).

---

### ➡️ BƯỚC 1: DATA INPUT
**NÓI:**
> *"Input của mô hình là ảnh RGB thô. Đi qua 3 bước Preprocessing:*
> 1. *`Resize(448)` $\rightarrow$ Ảnh về **$448 \times 448 \times 3$**.*
> 2. *`CenterCrop(392)` $\rightarrow$ Cắt vuông vùng giữa **$392 \times 392 \times 3$**. (Tại sao cắt? Để loại bỏ vùng viền ngoài nhiễu và khớp với kích thước patch $14 \times 14$ của ViT).*
> 3. *`Normalize` ImageNet mean/std $\rightarrow$ Tensor đưa vào mô hình shape **`[B, 3, 392, 392]`**, với `B` là batch size (16 khi train, 1 khi inference)."*

---

### ➡️ BƯỚC 2: DINOV2-REGISTER BACKBONE (FROZEN)
**CHỈ VÀO CODE — Hàm `extract_intermediate_features` trong [`vitill_gct.py` L69-93](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py#L69-L93):**
```python
# vitill_gct.py - hàm extract_intermediate_features()
outputs = backbone.get_intermediate_layers(
    x, n=target_layers, return_class_token=return_cls
)   # DINOv2 official API — trả về list tuple (patch_tokens, cls_token)

feat_list = [o[0] for o in outputs]   # 8 × [B, 784, 768]
cls_token  = outputs[-1][1]            # [B, 768] — CLS từ layer cuối cùng
```
**NÓI:**
> *"Backbone là DINOv2-Register ViT-B/14. Em **đóng băng toàn bộ** (`requires_grad=False`), không train lại.*
>
> *ViT chia ảnh thành các ô vuông Patch $14 \times 14$ pixel:*
> $$N = \frac{392}{14} \times \frac{392}{14} = 28 \times 28 = 784 \text{ patch tokens}$$
>
> *Em dùng API chính thức của DINOv2 là `backbone.get_intermediate_layers()` — truyền vào list `[2,3,4,5,6,7,8,9]` là chỉ số 8 lớp cần trích. API này trả về list tuple, mỗi tuple gồm `(patch_tokens, cls_token)` của lớp đó.*
>
> *`feat_list`: 8 tensor mỗi cái shape `[B, 784, 768]` — đặc trưng 8 lớp trung gian.*
> *`cls_token`: lấy từ `outputs[-1][1]` — CLS token của lớp sâu nhất (layer 9), shape `[B, 768]`. Em dùng CLS token này làm target để train GCT ạ."*

---

### ➡️ BƯỚC 3: BOTTLENECK MLP + GCT TOKEN INSERTION
**CHỈ VÀO CODE — Hàm `forward` trong [`ViTillGCT` L220-238](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py#L220-L238):**
```python
# vitill_gct.py - ViTillGCT.forward()
# Bước 1: Average-fuse toàn bộ 8 lớp encoder
x = self.fuse_features(feat_list, list(range(len(feat_list))))  # [B, 784, 768]

# Bước 2: Bottleneck MLP (768 -> 3072 -> 768)
x = self.bottleneck(x)  # [B, 784, 768]

# Bước 3: GCTModule.prepend() — ghép GCT Token vào đầu chuỗi
x = self.gct.prepend(x)  # [B, 785, 768]  (784 + 1 GCT)
```
**NÓI:**
> *"Em gộp 8 lớp đặc trưng lại bằng hàm `self.fuse_features()` — average toàn bộ 8 layer về 1 tensor `[B, 784, 768]`. Rồi đưa qua Bottleneck MLP:*
> $$768 \rightarrow 3072 \xrightarrow{\text{GELU}} 3072 \rightarrow 768$$
>
> *Đây là phần em thêm vào: `self.gct.prepend(x)` — bên trong hàm này, em lấy parameter `self.gct_token` shape `[1, 1, 768]`, expand theo batch thành `[B, 1, 768]`, rồi `torch.cat` ghép vào đầu:*
> $$[B, 1, 768] \oplus [B, 784, 768] \rightarrow [B, 785, 768]$$
>
> *Chuỗi `[B, 785, 768]` này đi vào 8 lớp Decoder."*

---

### ➡️ BƯỚC 4: TRANSFORMER DECODER (8 KHỐI LINEAR ATTENTION)
**CHỈ VÀO CODE — [`vitill_gct.py` L240-254](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py#L240-L254):**
```python
# vitill_gct.py - ViTillGCT.forward()
de_list = []
for blk in self.decoder:              # Lần lượt qua 8 DecoderBlock
    x = blk(x)                        # x: [B, 785, 768] -> [B, 785, 768]
    # Thu thập patch tokens (bỏ GCT ở position 0)
    de_list.append(x[:, 1:, :])       # [B, 784, 768] — chỉ lấy patch

# Sau vòng lặp, x[:, 0, :] là GCT token output của block CUỐI CÙNG
gct_final = x[:, 0, :]               # [B, 768]
gct_loss  = self.gct.compute_loss(gct_final, cls_token)
```
**NÓI:**
> *"Chuỗi `[B, 785, 768]` đi qua vòng lặp 8 khối `DecoderBlock`. Em dùng Linear Attention thay vì Softmax Attention chuẩn vì:*
> - *Softmax Attention: độ phức tạp $O(N^2)$ với $N = 785 \Rightarrow 616,225$ phép tính.*
> - *Linear Attention (ELU+1): độ phức tạp $O(N)$, nhanh hơn rất nhiều.*
>
> *Sau mỗi block, em thu thập `x[:, 1:, :]` — chỉ lấy 784 patch tokens, bỏ GCT token position 0 ra khỏi `de_list`.*
>
> *Sau khi chạy hết 8 blocks, `x[:, 0, :]` là decoded GCT token cuối cùng, shape `[B, 768]`. Em đưa token này vào `self.gct.compute_loss()` để tính GCT Loss ạ."*

---

## ▶️ PHẦN 3 (Phút 9:00 – 14:00): GCT LOSS PIPELINE — PHẦN MÌNH THIẾT KẾ

### 🖥️ SHOW GÌ:
> Mở file [`src/losses/cosine_loss.py`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/losses/cosine_loss.py) và [`src/models/vitill_gct.py`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py).

---

### 🔵 NHÁNH A — RECONSTRUCTION LOSS (Baseline cũ đã có)
**NÓI:**
> *"Loss tái tạo cục bộ ($\mathcal{L}_{\text{rec}}$): Với mỗi batch, em tính Cosine Distance giữa từng cặp patch Encoder/Decoder:*
> $$a(i) = 1.0 - \text{CosSim}(\mathbf{x}_{\text{enc}}^{(i)}, \mathbf{x}_{\text{dec}}^{(i)})$$
>
> *Sau đó lấy Top 10% patch khó nhất (Hard-mining ratio $p = 0.9$) để tính loss, tránh bị patch nền tĩnh dễ lấn át.*
> - *Tại sao Top 10%? Vì 90% patch trong ảnh sản phẩm bình thường nằm ở vùng nền rất dễ tái tạo. Chỉ vùng lỗi nhỏ mới khó — Hard-mining ép mô hình tập trung vào đó."*

---

### 🟠 NHÁNH B — GCT LOSS (Phần mình đề xuất)
**CHỈ VÀO CODE — [`GCTModule.compute_loss()` L135-147](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py#L135-L147):**
```python
# vitill_gct.py - GCTModule.compute_loss()
def compute_loss(self, gct_final, cls_token):
    proj_gct     = self.projection_head(gct_final)   # Linear(768,768)+LayerNorm -> [B, 768]
    cls_detached = cls_token.detach()                 # KHÔNG cho gradient vào DINOv2!
    return (1.0 - F.cosine_similarity(proj_gct, cls_detached, dim=-1)).mean()
```
**NÓI:**
> *"Đây là phần em thiết kế thêm. Decoded GCT Token `[B, 768]` đi qua `self.projection_head` — 1 lớp Linear + LayerNorm:*
> $$\mathbf{p}_{\text{gct}} = \text{LayerNorm}(\text{Linear}_{768 \rightarrow 768}(\mathbf{t}_{\text{gct}}^{\text{out}})) \in \mathbb{R}^{768}$$
>
> *Tại sao 1 lớp thay vì MLP 2 lớp như V1? Vì MLP 2 lớp bị **Head Capacity Shortcut** — gradient bị hút vào projection head, không truyền ngược được về 8 lớp Decoder. Với 1 lớp Linear + LayerNorm, gradient chảy thẳng về Decoder mượt mà hơn.*
>
> *GCT Loss là khoảng cách Cosine giữa vector GCT sau projection và CLS token đại cục của DINOv2:*
> $$\mathcal{L}_{\text{GCT}} = 1.0 - \text{CosSim}(\mathbf{p}_{\text{gct}}, \underbrace{\mathbf{c}_{\text{dino}}}_{\text{.detach()}})$$
>
> *`.detach()` ở `cls_token` là bắt buộc — nếu không gradient sẽ trôi về thay đổi trọng số DINOv2 đã đóng băng."*

---

### 🟢 TỔNG LOSS & GIẢI THÍCH CÔNG THỨC $\mathcal{L}_{	ext{total}}$
**NÓI:**
> *"Tổng Loss huấn luyện của mô hình:*
> $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rec}} + \lambda \cdot \mathcal{L}_{\text{GCT}}$$
>
> *Tại sao lại có công thức này và hệ số $\lambda$? Có 3 lý do kỹ thuật cốt lõi:*
> 1. ***Tối ưu đa mục tiêu (Multi-Task Learning):*** $\mathcal{L}_{\text{rec}}$ ép 8 lớp Decoder tái tạo chi tiết 784 patch cục bộ để bắt vết xước (Structural). $\mathcal{L}_{\text{GCT}}$ ép GCT Token học quy luật ngữ cảnh đại cục từ DINOv2 CLS token để bắt lỗi thiếu linh kiện (Logical).
> 2. ***Cân bằng Gradient với $\lambda$ ($\Delta \lambda = 0.5$):*** $\mathcal{L}_{\text{rec}}$ là trung bình trên 784 patch tokens (biên độ $0.01 \rightarrow 0.05$), còn $\mathcal{L}_{\text{GCT}}$ là khoảng cách của 1 vector đơn lẻ (biên độ $0.1 \rightarrow 0.6$). Trọng số $\lambda$ giúp cân bằng lực gradient, không để luồng GCT áp đảo luồng tái tạo patch.
> 3. ***Tại sao dùng Cosine Distance thay vì MSE/$L_2$?*** Trong không gian 768 chiều, khoảng cách Euclidean ($L_2$) dễ nổ gradient do độ lớn vector. Khoảng cách Cosine ($1 - \text{CosSim}$) chỉ đo góc lệch hướng, giúp gradient ổn định và hội tụ siêu nhanh ạ."*

---

### 🟣 GIẢI THÍCH CƠ CHẾ LAYER GROUPING (SO VỚI PAPER GỐC DINOMALY)
**NÓI (KHI CHỈ VÀO CODE `fuse_layer_enc` TRONG `vitill_gct.py`):**
> *"Về cơ chế Layer Grouping:*
> - *Paper gốc Dinomaly chia 8 lớp trung gian encoder thành 2 nhóm: `[0,1,2,3]` (nông - texture) và `[4,5,6,7]` (sâu - semantic).*
> - *Trong GCT V2 của em:*
>   + *Khi đưa qua Bottleneck, em cộng trung bình cả 8 lớp (`self.fuse_features(feat_list, range(8))`) để Bottleneck có góc nhìn toàn cục tốt nhất cho GCT Token.*
>   + *Nhưng khi tính Reconstruction Loss ($\mathcal{L}_{\text{rec}}$), em vẫn giữ nguyên 2 nhóm `fuse_layer_enc` và `fuse_layer_dec` theo đúng paper gốc để so sánh Encoder-Decoder theo từng cấp độ ngữ nghĩa nông/sâu ạ."*

---

### 🔴 INFERENCE — ACTIVE DUAL-STREAM SCORING
**CHỈ VÀO CODE trong [`src/eval.py` L108-134](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/eval.py#L108-L134):**
```python
# eval.py - infer_one()
en, de, gct_loss = model(feat_list, cls_token)
score_gct   = float(gct_loss.item())          # Global alignment score

amap        = compute_anomaly_map(en, de, crop_size=392, out_size=448)
score_patch = image_score(amap)             # Top-1% mean patch error

# GCT V2 Dual-Stream: kết hợp cục bộ + đại cục
score_final = score_patch + (gamma * score_gct if use_gct else 0.0)
```
**NÓI:**
> *"Lúc inference, em kết hợp 2 luồng điểm:*
> $$\text{Score}_{\text{final}} = \underbrace{\text{Score}_{\text{patch}}}_{\text{Cục bộ — patch}} + 1.0 \cdot \underbrace{\text{Score}_{\text{GCT}}}_{\text{Đại cục — toàn ảnh}}$$
>
> *`Score_patch`: Top-1% mean của Anomaly Map (tránh 1 pixel cực trị lừa điểm).*
>
> *`Score_GCT`: Chính là $\mathcal{L}_{\text{GCT}}$ — khoảng cách Cosine đại cục lúc inference. Ảnh bình thường thì $\text{Score}_{\text{GCT}} \approx 0.002$ (rất thấp). Ảnh lỗi logic thì bộc phát tăng vọt lên $\approx 0.14$.*
>
> *Hệ số $\gamma = 1.0$ được cố định a-priori — không chỉnh tay trên tập test để tránh Data Leakage."*

---

## ▶️ PHẦN 4 (Phút 14:00 – 17:00): KẾT QUẢ THỰC NGHIỆM

### 🖥️ SHOW GÌ:
> Mở file [`docs/presentation_summary.md`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/docs/presentation_summary.md) hoặc chiếu bảng sau:

| Category | Baseline Logical | GCT V2 Logical | Gain |
|:---|:---:|:---:|:---:|
| Breakfast Box | 70.25% | 73.41% | **+3.16%** |
| Juice Bottle | 90.33% | 91.76% | **+1.43%** |
| Pushpins | 86.15% | 88.23% | **+2.08%** |
| Screw Bag | 59.19% | **68.63%** | 🔥 **+9.44%** |
| Splicing Connectors | 76.08% | 79.62% | **+3.54%** |
| **Mean** | **76.40%** | **80.33%** | 🏆 **+3.93%** |

| Metric | Baseline | GCT V2 | Gain |
|:---|:---:|:---:|:---:|
| **Mean AUROC** | 84.67% | **86.68%** | **+2.01%** |
| **Logical AUROC** | 76.40% | **80.33%** | **+3.93%** |
| **Structural AUROC** | 92.97% | **93.04%** | **+0.07%** |
| **sPRO*** | 70.53% | **70.10%** | Duy trì |
| **Latency** | ~87ms | ~87ms | Real-time |

### 🗣️ NÓI GÌ:
> *"Đây là kết quả đánh giá trên 5 category của MVTec LOCO AD.*
>
> *Điểm đáng chú ý nhất: **Logical AUROC tăng +3.93%** (từ 76.40% lên 80.33%). Tập `screw_bag` — ốc vít rời — tăng kỷ lục **+9.44%**, đây là tập khó nhất vì lỗi thuần túy là đếm số lượng ốc sai — đúng trọng tâm bài toán Logical Anomaly mà GCT Token được thiết kế để giải quyết.*
>
> *Structural AUROC gần như giữ nguyên (92.97% $\rightarrow$ 93.04%) — đúng với kỳ vọng vì GCT không can thiệp vào luồng tái tạo patch cục bộ.*
>
> *Latency giữ nguyên ~87ms vì GCT Token chỉ thêm 1 token tính toán, không ảnh hưởng tốc độ."*

---

## ▶️ PHẦN 5 (Phút 17:00 – 20:00): BUG ĐÃ FIX + SPATIAL ALIGNMENT

### 🖥️ SHOW GÌ:
> Mở code phần `compute_anomaly_map` trong [`src/eval.py` L57-92](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/eval.py#L57-L92).

### 🗣️ NÓI GÌ:
> *"Em xin trình bày nhanh 2 bug kỹ thuật quan trọng nhất đã fix:*
>
> **Bug 1 — Gradient Leak:**
> Khi tính GCT Loss, ban đầu em quên `.detach()` CLS token. Gradient từ GCT Loss trôi ngược về thay đổi trọng số DINOv2 đang đóng băng $\rightarrow$ đặc trưng bị hỏng, loss không hội tụ. Fix: `cls_token.detach()`.
>
> **Bug 2 — Spatial Coordinate Mismatch ($392 \rightarrow 448$):**
>
> *Ảnh input $448 \times 448$ bị `CenterCrop` lấy $392 \times 392$. Anomaly Map từ Decoder chỉ cover vùng $392 \times 392$ giữa ảnh. Nếu upsample thẳng từ $28 \times 28$ lên $448 \times 448$, bản đồ bất thường bị kéo giãn và lệch tọa độ so với GT Mask $448 \times 448$:*

```python
# Fix trong eval.py compute_anomaly_map() L57-92:
# Upsample về crop size trước (392x392)
a_map = F.interpolate(a_map, size=(crop_size, crop_size), mode='bilinear')

# Dán vào canvas 448x448 ở vị trí chính xác (offset = 28px)
canvas = np.zeros((out_size, out_size))
top  = (out_size - crop_size) // 2   # = 28
left = (out_size - crop_size) // 2   # = 28
canvas[top:top + crop_size, left:left + crop_size] = crop_amap
```

> *Fix này giúp sPRO tăng từ 60.90% lên **70.10%** (+9.2%)."*

---

## ▶️ PHẦN PHỤ: ĐÓNG GÓP CỦA EM SO VỚI PAPER GỐC DINOMALY

> **Đây là câu Thầy CHẮC CHẮN sẽ hỏi:** *"Em đã thay đổi/đóng góp gì so với paper gốc?"*

### 🖥️ SHOW GÌ:
> Mở [`src/models/vitill_gct.py`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py) song song và chiếu bảng so sánh dưới đây.

---

### 📊 BẢNG SO SÁNH CHI TIẾT — PAPER GỐC vs GCT V2

| Thành Phần | Paper Gốc Dinomaly | ViTill-GCT V2 (của em) | Code |
|:---|:---|:---|:---|
| **Backbone** | DINOv2-Register ViT-B/14 (frozen) | DINOv2-Register ViT-B/14 (frozen) ✅ Giữ nguyên | [L1-66](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py#L1-L66) |
| **Feature trích xuất** | 8 layer qua `get_intermediate_layers()` | 8 layer qua `get_intermediate_layers()` ✅ Giữ nguyên | [L69-93](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py#L69-L93) |
| **Bottleneck** | `bMlp(768→3072→768, drop=0.2)` | `bMlp(768→3072→768, drop=0.2)` ✅ Giữ nguyên | [L187-193](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py#L187-L193) |
| **GCT Token** | ❌ Không có | ✅ **THÊM MỚI**: 1 learnable parameter `[1,1,768]`, prepend trước Decoder | [L116-133](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py#L116-L133) |
| **Decoder** | 8 × LinearAttention2 DecoderBlock | 8 × LinearAttention2 DecoderBlock ✅ Giữ nguyên (GCT đi xuyên qua cùng) | [L199-208](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py#L199-L208) |
| **Projection Head** | ❌ Không có | ✅ **THÊM MỚI**: `Linear(768→768) + LayerNorm` trong `GCTModule` | [L124-127](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py#L124-L127) |
| **GCT Loss** | ❌ Không có | ✅ **THÊM MỚI**: $1 - \text{CosSim}(\text{proj}(t_{\text{gct}}),\ \text{CLS}.\text{detach}())$ | [L135-147](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py#L135-L147) |
| **Loss tổng** | $\mathcal{L} = \mathcal{L}_{\text{rec}}$ | $\mathcal{L} = \mathcal{L}_{\text{rec}} + \lambda \cdot \mathcal{L}_{\text{GCT}}$ ✅ **MỞ RỘNG** | [L112-133](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/losses/cosine_loss.py#L112-L133) |
| **Inference Scoring** | `Score = Top-1% patch error` | ✅ **THÊM MỚI**: `Score_patch + γ × Score_GCT` (Dual-Stream) | [L108-134](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/eval.py#L108-L134) |
| **Spatial Alignment** | Không mô tả rõ | ✅ **BUG FIX**: upsample `392×392` $\rightarrow$ paste canvas `448×448` offset=28 | [L57-92](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/eval.py#L57-L92) |

---

### 🔎 CODE MAPPING 3 ĐÓNG GÓP CHÍNH

#### ① GCT Token + Projection Head — [`vitill_gct.py` L116-147](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py#L116-L147)
```python
# class GCTModule — vitill_gct.py L116-147
self.gct_token = nn.Parameter(torch.zeros(1, 1, embed_dim))   # [1,1,768] learnable
nn.init.trunc_normal_(self.gct_token, std=0.01)

self.projection_head = nn.Sequential(
    nn.Linear(embed_dim, embed_dim),   # 768 → 768
    nn.LayerNorm(embed_dim),           # Normalize
)

def prepend(self, x):
    gct = self.gct_token.expand(B, -1, -1)   # [B, 1, 768]
    return torch.cat([gct, x], dim=1)         # [B, 785, 768]

def compute_loss(self, gct_final, cls_token):
    proj_gct     = self.projection_head(gct_final)   # [B, 768]
    cls_detached = cls_token.detach()                 # NO gradient to DINOv2!
    return (1.0 - F.cosine_similarity(proj_gct, cls_detached, dim=-1)).mean()
```

#### ② Dual-Stream Scoring — [`eval.py` L108-134](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/eval.py#L108-L134)
```python
# infer_one() — eval.py L108-134
en, de, gct_loss = model(feat_list, cls_token)
score_gct = float(gct_loss.item())          # Global alignment score

amap        = compute_anomaly_map(en, de, crop_size=392, out_size=448)
score_patch = image_score(amap)             # Top-1% mean patch error

# GCT V2 Dual-Stream: kết hợp cục bộ + đại cục
score_final = score_patch + (gamma * score_gct if use_gct else 0.0)
```

#### ③ Spatial Alignment Bug Fix — [`eval.py` L57-92](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/eval.py#L57-L92)
```python
# compute_anomaly_map() — eval.py L57-92
# Upsample về CROP SIZE (392) trước — không phải out_size (448)!
a_map = F.interpolate(a_map, size=(crop_size, crop_size), mode='bilinear')

# Paste vào canvas out_size x out_size tại đúng offset = (448-392)//2 = 28
canvas = np.zeros((out_size, out_size))
top  = (out_size - crop_size) // 2   # = 28
left = (out_size - crop_size) // 2   # = 28
canvas[top:top + crop_size, left:left + crop_size] = crop_amap
# sPRO: 60.90% → 70.10% (+9.2%)
```

---

### 🗣️ NÓI GÌ KHI THẦY HỎI "EM ĐÃ ĐÓNG GÓP GÌ?":

> *"Dạ Thầy, em giữ nguyên toàn bộ kiến trúc backbone và decoder của Dinomaly gốc. Phần đóng góp của em gồm **3 điểm chính**:*
>
> **1. GCT Token + Projection Head + GCT Loss** — [`vitill_gct.py` L116-147](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py#L116-L147):
> *Em chèn 1 `nn.Parameter` học được shape `[1,1,768]` vào đầu chuỗi trước Decoder. Token này tương tác Attention với 784 patch token xuyên suốt 8 lớp. Sau decoder, em lấy `x[:, 0, :]` — GCT token output cuối — đưa qua Projection Head rồi tính Cosine Distance với CLS token DINOv2 (`.detach()`). Loss này buộc Decoder học quy luật ngữ cảnh đại cục.*
>
> **2. Dual-Stream Active Scoring** — [`eval.py` L108-134](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/eval.py#L108-L134):
> *Tại inference, `score_final = score_patch + γ × score_gct`. Ảnh lỗi logic thì `score_gct` tăng vọt dù `score_patch` thấp — đây là cơ chế bắt được Logical Anomaly mà Dinomaly gốc không làm được.*
>
> **3. Spatial Alignment Bug Fix** — [`eval.py` L57-92](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/eval.py#L57-L92):
> *Upsample Anomaly Map về đúng $392 \times 392$ rồi paste vào canvas $448 \times 448$ tại offset 28px. Fix này giúp sPRO tăng +9.2%."*

---

## 💡 GHI CHÚ KHI THUYẾT TRÌNH

> [!IMPORTANT]
> **KHÔNG CẦN GIỚI THIỆU** "tại sao làm đề tài này", "MVTec LOCO AD là gì", "ứng dụng nhà máy"... Thầy đã biết hết rồi. Đi thẳng vào Model từ đầu.

> [!TIP]
> Khi chỉ vào sơ đồ kiến trúc hoặc code, dùng ngón tay hoặc con trỏ chuột chỉ đúng vào phần đang nói. Không nhìn vào màn hình quá lâu — nhìn Thầy và nói.

> [!TIP]
> Nếu Thầy ngắt và hỏi giữa chừng $\rightarrow$ Dừng ngay, trả lời trực tiếp, sau đó tiếp tục đúng chỗ đang dở.

> [!CAUTION]
> Câu nói an toàn khi bị hỏi khó: *"Dạ, đây là một điểm rất thú vị Thầy ạ. Em hiểu nguyên lý chung là... [nói phần biết]. Phần chi tiết hơn em sẽ tiếp tục phân tích và báo cáo Thầy ở buổi Cuối kỳ ạ."*
