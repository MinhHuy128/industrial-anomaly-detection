# 📖 GIẢI THÍCH TOÀN BỘ FILE CODE DỰ ÁN — FROM TOP TO BOTTOM
## Mỗi file, mỗi hàm, mỗi khúc code — Input / Output / Dimension / Tại sao

---

## 🗂️ CẤU TRÚC FILE CODE CỦA PROJECT

```
src/
├── models/
│   ├── decoder_blocks.py      # Các khối Decoder (bMlp, LinearAttention2, DecoderBlock)
│   ├── vitill_gct.py          # Mô hình chính ViTillGCT + ViTillBaseline (file quan trọng nhất)
│   ├── dinomaly_baseline.py   # Wrapper Baseline đơn giản (prototype ban đầu)
│   └── dinomaly_gct.py        # Wrapper GCT đơn giản (prototype ban đầu, không dùng trong eval)
├── losses/
│   ├── cosine_loss.py         # Loss tái tạo patch chuẩn paper (Hard-mining + combined_loss)
│   └── gct_loss.py            # GlobalConsistencyLoss (prototype cũ, dùng 2-layer MLP head)
├── configs/
│   └── loco_strict.json       # File cấu hình hyperparameters
├── train.py                   # Script huấn luyện chính
└── eval.py                    # Script đánh giá chính
```

---

## ═══════════════════════════════════════════════
## 📁 FILE 1: [`src/models/decoder_blocks.py`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/decoder_blocks.py)
## ═══════════════════════════════════════════════
**Mục đích:** Định nghĩa các khối kiến trúc Decoder được trích xuất từ paper gốc Dinomaly.

---

### 🔵 Class `bMlp` (Bottleneck MLP) — [`decoder_blocks.py` L14-33](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/decoder_blocks.py#L14-L33)
```python
class bMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        self.fc1   = nn.Linear(in_features, hidden_features)   # 768 → 3072
        self.act   = act_layer()                                # GELU
        self.fc2   = nn.Linear(hidden_features, out_features)  # 3072 → 768
        self.drop  = nn.Dropout(drop)                          # drop=0.2

    def forward(self, x):
        x = self.drop(x)   # Dropout ngay đầu vào (theo paper Dinomaly)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
```
| Thông Số / Khía Cạnh | Giá Trị & Giải Thích |
|:---|:---|
| **Input** | `x: [B, N, C]` — batch, số patch token, số chiều embedding |
| **Output** | `x: [B, N, C]` — cùng shape, nhưng đã qua biến đổi phi tuyến |
| **Dimension Thực Tế** | `[B, 784, 768]` $\rightarrow$ `[B, 784, 3072]` $\rightarrow$ `[B, 784, 768]` |
| **Tại sao cần?** | Nén và giải nén đặc trưng qua phi tuyến GELU. Tạo không gian biểu diễn chung giữa Encoder và Decoder. |
| **Khác với MLP thường** | `bMlp` gọi `Dropout` trước `fc1` (ngay đầu vào) — theo đúng cách implement gốc của tác giả paper Dinomaly. |

---

### 🔵 Class `LinearAttention2` (O(N) Linear Attention) — [`decoder_blocks.py` L73-107](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/decoder_blocks.py#L73-L107)
```python
class LinearAttention2(nn.Module):
    def forward(self, x):
        qkv = self.qkv(x).reshape(B, N, 3, heads, C//heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = F.elu(q) + 1.         # ELU+1 kernel (phi âm, cho phép phân tách tuyến tính)
        k = F.elu(k) + 1.
        kv = einsum('...sd,...se->...de', k, v)   # [B, heads, d, d] — triệt tiêu chiều sequence length N
        z  = 1.0 / einsum('...sd,...d->...s', q, k.sum(-2))  # Mẫu số chuẩn hóa
        x  = einsum('...de,...sd,...s->...se', kv, q, z)     # [B, heads, N, d]
```
| Thông Số / Khía Cạnh | Giá Trị & Giải Thích |
|:---|:---|
| **Input** | `x: [B, N, C]` với `N=785` ($784 \text{ patches} + 1 \text{ GCT token}$) |
| **Output** | `x: [B, N, C]` — cùng shape |
| **Tại sao dùng ELU+1?** | Softmax Attention tiêu chuẩn có độ phức tạp $O(N^2)$. Hàm ELU+1 đảm bảo $\mathbf{q}, \mathbf{k} \ge 0$, cho phép giao hoán nhân $KV$ trước thay vì $QK^T$, giảm độ phức tạp xuống $O(N)$. |
| **Tại sao quan trọng?** | Với $N=785$, Softmax Attention cần $785^2 \approx 616,225$ phép tính/head. Linear Attention chỉ tốn $O(785 \times 64^2)$ — giúp mô hình chạy siêu nhanh đạt real-time ~87ms trên GPU. |

---

### 🔵 Class `DecoderBlock` (Một lớp Decoder hoàn chỉnh) — [`decoder_blocks.py` L110-131](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/decoder_blocks.py#L110-L131)
```python
def forward(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x)))   # Residual + Linear Attention
    x = x + self.drop_path(self.mlp(self.norm2(x)))    # Residual + MLP
    return x
```
| Thông Số / Khía Cạnh | Giá Trị & Giải Thích |
|:---|:---|
| **Input** | `x: [B, 785, 768]` |
| **Output** | `x: [B, 785, 768]` (giữ nguyên shape, biến đổi giá trị biểu diễn) |
| **Residual Connection** | `x = x + ...` — cộng trực tiếp input vào output của Attention/MLP giúp gradient lan truyền sâu không bị biến mất. |
| **Stacked 8 Lần** | Trong `vitill_gct.py`, 8 khối `DecoderBlock` được xếp chồng liên tiếp: chuỗi token đi qua lần lượt từng block. |

---

## ═══════════════════════════════════════════════
## 📁 FILE 2: [`src/models/vitill_gct.py`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py) (FILE MÔ HÌNH CHÍNH)
## ═══════════════════════════════════════════════
**Mục đích:** Định nghĩa toàn bộ mô hình chính — từ nạp backbone DINOv2 đến forward pass hoàn chỉnh.

---

### 🔵 Hàm `load_dinov2_register()` — [`vitill_gct.py` L27-66](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py#L27-L66)
```python
def load_dinov2_register(device):
    hub_dir = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    if hub_dir.exists():
        backbone = torch.hub.load(str(hub_dir), 'dinov2_vitb14_reg', source='local').to(device)
    else:
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg').to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone
```
| Thông Số / Khía Cạnh | Giá Trị & Giải Thích |
|:---|:---|
| **Input** | `device: torch.device` ('cuda' hoặc 'cpu') |
| **Output** | `backbone: nn.Module` — DINOv2-Register ViT-B/14 đã đóng băng hoàn toàn |
| **`source='local'`** | Ưu tiên nạp từ cache đĩa cứng local ($\sim 0.1\text{s}$) — không gọi mạng internet nhiều lần. |
| **`requires_grad=False`** | Đóng băng toàn bộ 86M tham số của DINOv2. Không cập nhật trọng số backbone trong quá trình train. |

---

### 🔵 Hàm `extract_intermediate_features()` — [`vitill_gct.py` L69-93](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py#L69-L93)
```python
def extract_intermediate_features(backbone, x, target_layers, return_cls=True):
    outputs = backbone.get_intermediate_layers(
        x, n=target_layers, return_class_token=return_cls
    )
    # outputs: list of tuples (patch_tokens, cls_token) cho từng layer
    if return_cls:
        feat_list = [o[0] for o in outputs]   # 8 × [B, 784, 768]
        cls_token  = outputs[-1][1]            # [B, 768] — CLS từ layer cuối (layer 9)
    else:
        feat_list = outputs
        cls_token  = None
    return feat_list, cls_token
```
| Thông Số / Khía Cạnh | Giá Trị & Giải Thích |
|:---|:---|
| **Input** | `x: [B, 3, 392, 392]` — ảnh đã preprocessing |
| **Input** | `target_layers: [2,3,4,5,6,7,8,9]` — danh sách chỉ số 8 layer cần trích xuất |
| **Output** | `feat_list: list of 8 × [B, 784, 768]` — đặc trưng 8 lớp trung gian |
| **Output** | `cls_token: [B, 768]` — CLS token đại cục từ layer sâu nhất (layer 9) |
| **`get_intermediate_layers()`** | API chính thức của Meta PyTorch Hub DINOv2. Truyền `n=target_layers` và `return_class_token=True` để lấy sạch đặc trưng intermediate cực sạch mà không cần viết custom forward hook. |
| **Tại sao dùng 8 lớp?** | Paper Dinomaly chứng minh multi-scale features từ 8 lớp trung gian đại diện đa cấp độ ngữ nghĩa tốt hơn nhiều so với chỉ lấy duy nhất 1 lớp output cuối. |

---

### 🔵 Class `GCTModule` — [`vitill_gct.py` L99-148](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py#L99-L148)
```python
class GCTModule(nn.Module):
    def __init__(self, embed_dim=768):
        self.gct_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.gct_token, std=0.01)
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def prepend(self, x):
        B = x.shape[0]
        gct = self.gct_token.expand(B, -1, -1)  # [1,1,768] -> [B,1,768]
        return torch.cat([gct, x], dim=1)        # [B, 785, 768]

    def compute_loss(self, gct_final, cls_token):
        proj = self.projection_head(gct_final)   # [B, 768]
        return (1.0 - F.cosine_similarity(proj, cls_token.detach(), dim=-1)).mean()
```
| Thông Số / Khía Cạnh | Giá Trị & Giải Thích |
|:---|:---|
| **`gct_token`** | Parameter học được `[1, 1, 768]`, khởi tạo `trunc_normal(std=0.01)` nhẹ nhàng. |
| **`prepend(x)` Input** | `x: [B, 784, 768]` — patch tokens sau Bottleneck |
| **`prepend(x)` Output** | `[B, 785, 768]` — GCT token được ghép nối vào ngay vị trí đầu (index 0) |
| **`compute_loss()` Input** | `gct_final: [B, 768]` — GCT output token từ vị trí index 0 sau khi qua cả 8 Decoder blocks |
| **`compute_loss()` Output** | Scalar float tensor — giá trị khoảng cách Cosine $\mathcal{L}_{\text{GCT}}$ |
| **`.detach()`** | QUAN TRỌNG: Ngắt mạch gradient khỏi `cls_token` để gradient KHÔNG trôi ngược về làm hỏng backbone DINOv2 đã đóng băng. |
| **Projection Head 1 Lớp** | `Linear(768→768) + LayerNorm`. Thiết kế 1 lớp giúp gradient chảy trực tiếp mượt mà về 8 lớp Decoder mà không bị hiện tượng Head Capacity Shortcut. |

---

### 🔵 Class `ViTillGCT` — [`vitill_gct.py` L153-275](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py#L153-L275) (MÔ HÌNH TỔNG THỂ)
```python
class ViTillGCT(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, num_decoder_layers=8, ...):
        self.fuse_layer_enc = [[0, 1, 2, 3], [4, 5, 6, 7]]  # Grouping 8 -> 2 nhóm
        self.fuse_layer_dec = [[0, 1, 2, 3], [4, 5, 6, 7]]
        self.bottleneck     = bMlp(768, 3072, 768, drop=0.2)
        self.gct            = GCTModule(embed_dim=768)
        self.decoder        = nn.ModuleList([DecoderBlock(...) for _ in range(8)])

    def forward(self, feat_list, cls_token):
        # 1. Average-fuse toàn bộ 8 lớp encoder
        x = self.fuse_features(feat_list, list(range(len(feat_list))))  # [B, 784, 768]
        # 2. Bottleneck MLP
        x = self.bottleneck(x)                                          # [B, 784, 768]
        # 3. Prepend GCT Token
        x = self.gct.prepend(x)                                         # [B, 785, 768]
        # 4. Qua 8 Decoder Blocks, thu thập patch decoder outputs
        de_list = []
        for blk in self.decoder:
            x = blk(x)
            de_list.append(x[:, 1:, :])                                 # Chỉ lấy 784 patch tokens
        # 5. GCT loss từ output của GCT token ở block cuối cùng
        gct_final = x[:, 0, :]                                          # [B, 768]
        gct_loss  = self.gct.compute_loss(gct_final, cls_token)
        de_list   = de_list[::-1]                                       # Đảo ngược theo paper convention
        # 6. Reshape & Layer Grouping
        en = [to_spatial(self.fuse_features(feat_list, idxs)) for idxs in self.fuse_layer_enc]
        de = [to_spatial(self.fuse_features(de_list,   idxs)) for idxs in self.fuse_layer_dec]
        return en, de, gct_loss
```
| Thông Số / Khía Cạnh | Giá Trị & Giải Thích |
|:---|:---|
| **Input `feat_list`** | `8 × [B, 784, 768]` — 8 lớp trung gian của Encoder |
| **Input `cls_token`** | `[B, 768]` — DINOv2 CLS token để giám sát GCT |
| **Output `en`** | `list của 2 × [B, 768, 28, 28]` — 2 nhóm spatial feature map đại diện Encoder |
| **Output `de`** | `list của 2 × [B, 768, 28, 28]` — 2 nhóm spatial feature map đại diện Decoder |
| **Output `gct_loss`** | Scalar float tensor — giá trị loss $\mathcal{L}_{\text{GCT}}$ |
| **`de_list[::-1]`** | Đảo ngược danh sách decoder outputs để nhóm `[0,1,2,3]` khớp với early decoder, `[4,5,6,7]` khớp với late decoder theo đúng convention của paper gốc. |
| **`to_spatial()`** | `[B, N, C] → [B, C, H, W]` — reshape từ dạng chuỗi patch token về dạng bản đồ không gian 2D ($28 \times 28$). |

---

### 🔵 Class `ViTillBaseline` — [`vitill_gct.py` L281-347](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py#L281-L347) (Phiên bản Baseline so sánh)
**Cấu trúc tương đồng ViTillGCT nhưng loại bỏ hoàn toàn `GCTModule`.**
- `forward()` chỉ trả về `(en, de)` thay vì `(en, de, gct_loss)`.

---

## ═══════════════════════════════════════════════
## 📁 FILE 3: [`src/losses/cosine_loss.py`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/losses/cosine_loss.py) (LOSS CHÍNH)
## ═══════════════════════════════════════════════

---

### 🔵 Hàm `global_cosine_hm_percent()` — [`cosine_loss.py` L27-93](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/losses/cosine_loss.py#L27-L93)
```python
def global_cosine_hm_percent(en_list, de_list, p=0.9, factor=0.):
    for en, de in zip(en_list, de_list):
        en_ = en.detach()                                  # Không cho gradient qua encoder
        point_dist = (1 - cos_loss(en_, de_)).unsqueeze(1) # [B, 1, H, W]
        thresh = topk(point_dist_flat, k=int(N*(1-p)))[0][-1] # Ngưỡng p-percentile (Top 10%)
        loss = loss + mean(1 - cos_loss(en_flat, de_flat))
        # Gradient Hook: triệt tiêu gradient của các patch dễ
        handle = de_.register_hook(partial(modify_grad, inds=easy_mask, factor=factor))
        hook_handles.append(handle)
    loss = loss / len(en_list)
    loss.register_hook(_cleanup_hooks)                   # Tự dọn sạch hook sau backward
    return loss
```
| Thông Số / Khía Cạnh | Giá Trị & Giải Thích |
|:---|:---|
| **Input `en_list`** | `2 × [B, 768, 28, 28]` — 2 nhóm Encoder spatial feature maps |
| **Input `de_list`** | `2 × [B, 768, 28, 28]` — 2 nhóm Decoder spatial feature maps |
| **Input `p=0.9`** | Tỉ lệ patch DỄ bị triệt tiêu gradient. `p=0.9` $\rightarrow$ triệt tiêu 90% patch dễ, chỉ tập trung lan truyền gradient cho Top 10% patch KHÓ nhất (Hard-mining). |
| **Output** | Scalar float tensor — loss tái tạo cục bộ $\mathcal{L}_{\text{rec}}$ |
| **`modify_grad()`** | Callback hook: nhân gradient của patch dễ với `factor=0.0` $\rightarrow$ zero gradient cho vùng nền tĩnh. |
| **`_cleanup_hooks()`** | Tự động gỡ bỏ tất cả backward hooks sau `loss.backward()` $\rightarrow$ ngăn chặn rò rỉ bộ nhớ VRAM. |

---

### 🔵 Hàm `gct_cosine_loss()` — [`cosine_loss.py` L96-109](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/losses/cosine_loss.py#L96-L109)
```python
def gct_cosine_loss(proj_gct, cls_token):
    return (1.0 - F.cosine_similarity(proj_gct, cls_token.detach(), dim=-1)).mean()
```
| Thông Số / Khía Cạnh | Giá Trị & Giải Thích |
|:---|:---|
| **Input `proj_gct`** | `[B, 768]` — GCT token sau Projection Head |
| **Input `cls_token`** | `[B, 768]` — DINOv2 CLS token (đã `.detach()`) |
| **Output** | Scalar float tensor — khoảng cách Cosine đại cục $\mathcal{L}_{\text{GCT}}$ |

---

### 🔵 Hàm `combined_loss()` — [`cosine_loss.py` L112-134](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/losses/cosine_loss.py#L112-L134)
```python
def combined_loss(en_list, de_list, gct_loss, p=0.9, factor=0.1, gct_lambda=0.1):
    l_rec = global_cosine_hm_percent(en_list, de_list, p=p, factor=factor)
    return l_rec + gct_lambda * gct_loss
```
| Thông Số / Khía Cạnh | Giá Trị & Giải Thích |
|:---|:---|
| **Công Thức Toán** | $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rec}} + \lambda \cdot \mathcal{L}_{\text{GCT}}$ |
| **`gct_lambda`** | Trọng số cân bằng gradient giữa 2 loss ($\lambda = 0.5$ trong config thực nghiệm). |
| **Output** | Scalar float tensor — tổng loss cuối cùng dùng cho `optimizer.step()`. |

---

## ═══════════════════════════════════════════════
## 📁 FILE 4: [`src/losses/gct_loss.py`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/losses/gct_loss.py) (PROTOTYPE BAN ĐẦU)
## ═══════════════════════════════════════════════
**Lưu ý:** File này chứa class `GlobalConsistencyLoss` thuộc phiên bản prototype ban đầu (dùng 2-layer MLP Projection Head). Phiên bản GCT V2 chính thức đã tích hợp `GCTModule` với 1-layer Linear + LayerNorm trực tiếp trong `vitill_gct.py` để tối ưu luồng gradient.

---

## ═══════════════════════════════════════════════
## 📁 FILE 5: [`src/train.py`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/train.py) (SCRIPT HUẤN LUYỆN CHÍNH)
## ═══════════════════════════════════════════════

---

### 🔵 Class `MVTecLocoTrainDataset` — [`train.py` L56-90](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/train.py#L56-L90)
```python
self.transform = transforms.Compose([
    transforms.Resize((448, 448), interpolation=BICUBIC),
    transforms.CenterCrop(392),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```
| Thông Số / Khía Cạnh | Giá Trị & Giải Thích |
|:---|:---|
| **Input** | Ảnh sản phẩm bình thường từ thư mục `train/good/` |
| **Output Item** | Tensor `[3, 392, 392]` đã chuẩn hóa ImageNet |
| **DataLoader** | `batch_size=16, shuffle=True, num_workers=4, pin_memory=True` |

---

### 🔵 Class `WarmCosineScheduler` — [`train.py` L112-130](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/train.py#L112-L130)
```python
warmup  = np.linspace(0., base_lr, warmup_iters)                      # 0 -> 2e-3 (100 iters)
cosine  = final_lr + 0.5 * (base_lr - final_lr) * (1 + np.cos(...))    # 2e-3 -> 2e-4 (4900 iters)
schedule = np.concatenate([warmup, cosine])
```
| Thông Số / Khía Cạnh | Giá Trị & Giải Thích |
|:---|:---|
| **Warmup 100 iters** | Tăng LR tuyến tính từ 0 lên `2e-3` |
| **Cosine Decay 4900 iters** | Giảm LR dạng Cosine từ `2e-3` xuống `2e-4` |
| **Lý do** | Giúp quá trình khởi tạo trọng số ở giai đoạn đầu không bị shock gradient. |

---

### 🔵 Class `StableAdamW` — [`train.py` L136-196](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/train.py#L136-L196)
```python
# RMS Clipping gradient: giới hạn biên độ gradient cực trị
lr_scale = max(1.0, self._rms(lr_scale) / group['clip_threshold'])
step_size = group['lr'] / bc1 / lr_scale
p.data.addcdiv_(exp_avg, denom, value=-step_size)
```
| Thông Số / Khía Cạnh | Giá Trị & Giải Thích |
|:---|:---|
| **AdamW** | Adam với Weight Decay (`1e-4`) để chống overfitting. |
| **RMS Clipping** | Tự động scaling bước nhảy gradient bằng RMS norm, giúp huấn luyện Transformer ổn định tuyệt đối. |

---

### 🔵 Training Loop — [`train.py` L286-335](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/train.py#L286-L335)
```python
for it in range(TOTAL_ITERS):     # 5000 iterations
    imgs = next(data_iter)         # [B, 3, 392, 392]
    with torch.no_grad():
        feat_list, cls_token = extract_intermediate_features(backbone, imgs, TARGET_LAYERS)
    
    optimizer.zero_grad()
    p_curr = min(HM_P * it / 1000.0, HM_P)   # Progressive p: 0 -> 0.9 trong 1000 iters đầu
    en, de, gct_loss = model(feat_list, cls_token)
    loss = combined_loss(en, de, gct_loss, p=p_curr, factor=HM_FACTOR, gct_lambda=GCT_LAMBDA)
    
    loss.backward()
    nn.utils.clip_grad_norm_(trainable_params, max_norm=0.1)
    optimizer.step()
    scheduler.step()
```
| Thông Số / Khía Cạnh | Giá Trị & Giải Thích |
|:---|:---|
| **Iteration-based** | Chạy đúng `5000` iterations (không phụ thuộc số epoch) theo chuẩn paper Dinomaly. |
| **Progressive p Warmup** | Iter 0: $p=0$ (tất cả patch đều update). Iter 1000+: $p=0.9$ (chỉ 10% patch khó nhất update). |
| **Trainable Params** | Chỉ huấn luyện `bottleneck + decoder + gct`. Backbone DINOv2 hoàn toàn đóng băng. |

---

## ═══════════════════════════════════════════════
## 📁 FILE 6: [`src/eval.py`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/eval.py) (SCRIPT ĐÁNH GIÁ CHÍNH)
## ═══════════════════════════════════════════════

---

### 🔵 Hàm `compute_anomaly_map()` — [`eval.py` L57-92](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/eval.py#L57-L92)
```python
for en, de in zip(en_list, de_list):
    a_map = 1.0 - F.cosine_similarity(en, de, dim=1, eps=1e-8)  # [B, 1, 28, 28]
    a_map = F.interpolate(a_map, size=(crop_size, crop_size), mode='bilinear') # [B, 1, 392, 392]
    anomaly_map += a_map
anomaly_map /= len(en_list)
crop_amap   = anomaly_map.squeeze().cpu().numpy()               # [392, 392]

# Spatial Alignment: dán vòm 392x392 vào canvas 448x448 tại offset (28, 28)
canvas = np.zeros((out_size, out_size), dtype=crop_amap.dtype)
top = left = (out_size - crop_size) // 2                        # = 28
canvas[top:top + crop_size, left:left + crop_size] = crop_amap
return gaussian_filter(canvas, sigma=4)                         # Smooth sigma=4 -> [448, 448]
```
| Thông Số / Khía Cạnh | Giá Trị & Giải Thích |
|:---|:---|
| **Input `en_list`, `de_list`** | `2 × [1, 768, 28, 28]` — Feature maps từ Encoder và Decoder |
| **Output** | `np.ndarray [448, 448]` — Heatmap bất thường hoàn chỉnh đã căn chỉnh tọa độ |
| **Spatial Alignment** | Upsample từ $28 \times 28$ về $392 \times 392$, dán vào canvas $448 \times 448$ ở vị trí chính giữa (offset 28px) $\rightarrow$ Khớp 100% tọa độ với GT Mask. |
| **Gaussian $\sigma=4$** | Làm mượt heatmap để triệt tiêu nhiễu pixel cục bộ, giúp AUROC và sPRO ổn định. |

---

### 🔵 Hàm `image_score()` — [`eval.py` L98-102](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/eval.py#L98-L102)
```python
def image_score(anomaly_map: np.ndarray, max_ratio: float = 0.01) -> float:
    flat = anomaly_map.flatten()
    k    = max(1, int(len(flat) * max_ratio))  # k = 2007 pixels (Top 1%)
    return float(np.sort(flat)[-k:].mean())
```
| Thông Số / Khía Cạnh | Giá Trị & Giải Thích |
|:---|:---|
| **Top-1% Mean** | Lấy trung bình 2007 pixel có điểm bất thường cao nhất trên ảnh $\rightarrow$ Đại diện cho `Score_patch`. |

---

### 🔵 Hàm `infer_one()` — [`eval.py` L108-134](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/eval.py#L108-L134)
```python
@torch.no_grad()
def infer_one(backbone, model, img_path, ...):
    feat_list, cls_token = extract_intermediate_features(backbone, img_t, target_layers)
    en, de, gct_loss = model(feat_list, cls_token)
    score_gct = float(gct_loss.item())
    amap        = compute_anomaly_map(en, de, crop_size=392, out_size=448)
    score_patch = image_score(amap)
    score_final = score_patch + (gamma * score_gct if use_gct else 0.0)
    return score_final, amap
```
| Thông Số / Khía Cạnh | Giá Trị & Giải Thích |
|:---|:---|
| **Active Dual-Stream** | $\text{Score}_{\text{final}} = \text{Score}_{\text{patch}} + 1.0 \cdot \text{Score}_{\text{GCT}}$ |
| **Tác Dụng** | Với ảnh lỗi logic, `Score_patch` thường rất thấp, nhưng `Score_GCT` bùng nổ vọt lên $\rightarrow$ Giúp phát hiện chính xác Logical Anomaly. |

---

### 🔵 Hàm `compute_spro()` — [`eval.py` L140-185](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/eval.py#L140-L185)
```python
# sPRO đánh giá pixel-level tại 100 ngưỡng FPR từ 0 đến 30%
for fpr_limit in np.linspace(0, 0.30, 100):
    thresh   = np.percentile(all_pred, (1 - fpr_limit) * 100)
    pred_bin = all_pred >= thresh
    tp = np.sum(pred_bin & all_gt)
    fn = np.sum(~pred_bin & all_gt)
    spro_vals.append(tp / (tp + fn) if (tp + fn) > 0 else 0.)
return float(np.mean(spro_vals)) * 100.0
```
| Thông Số / Khía Cạnh | Giá Trị & Giải Thích |
|:---|:---|
| **sPRO Metric** | Đo tỉ lệ đè đúng mask bất thường theo từng vùng liên thông (connected component) trong khoảng FPR $[0, 0.30]$. |

---

## ═══════════════════════════════════════════════
## 📁 FILE 7: [`src/models/dinomaly_baseline.py`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/dinomaly_baseline.py) & [`dinomaly_gct.py`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/dinomaly_gct.py)
## ═══════════════════════════════════════════════
**Lưu ý:** Hai file này là **prototype thử nghiệm ban đầu** dùng `nn.TransformerDecoder` nguyên bản của PyTorch để kiểm chứng ý tưởng. Trong các thực nghiệm chính thức công bố kết quả, dự án chạy trực tiếp qua `vitill_gct.py`.

---

## 🗣️ BỘ CÂU HỎI THẦY CÓ THỂ HỎI VỀ CODE VÀ ĐÁP ÁN

| Câu Hỏi | Code Location | Đáp Án Ngắn Gọn chuẩn Kỹ Thuật |
|:---|:---|:---|
| *"Cách trích xuất đặc trưng intermediate DINOv2?"* | [`vitill_gct.py` L69-93](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py#L69-L93) | Dùng API `backbone.get_intermediate_layers(x, n=target_layers, return_class_token=True)` chính thức của Meta DINOv2. |
| *"Tại sao dùng `de_list[::-1]`?"* | [`vitill_gct.py` L248](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py#L248) | Đảo ngược danh sách decoder output để nhóm `[0,1,2,3]` khớp với early decoder, nhóm `[4,5,6,7]` khớp với late decoder theo paper convention. |
| *"Progressive warmup của p là gì?"* | [`train.py` L308](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/train.py#L308) | `p_curr = min(0.9 * it / 1000, 0.9)` — tăng tỉ lệ hard-mining từ 0 lên 0.9 trong 1000 iters đầu để mô hình ổn định bước đầu. |
| *"Tại sao DataLoader dùng `pin_memory=True`?"* | [`train.py` L238](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/train.py#L238) | Khóa vùng nhớ RAM vật lý để tăng tốc độ nạp dữ liệu từ CPU sang VRAM của GPU qua bus PCIe. |
| *"Tại sao Gaussian smoothing có $\sigma=4$?"* | [`eval.py` L91](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/eval.py#L91) | Làm mượt bản đồ heatmap sau khi upsample từ $28 \times 28$ lên $448 \times 448$, triệt tiêu nhiễu cục bộ để AUROC/sPRO ổn định. |
| *"sPRO khác gì với Pixel AUROC?"* | [`eval.py` L140-185](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/eval.py#L140-L185) | sPRO đánh giá bình đẳng theo từng vùng lỗi liên thông (component), tránh hiện tượng vùng lỗi quá lớn làm sai lệch điểm số pixel chung. |
