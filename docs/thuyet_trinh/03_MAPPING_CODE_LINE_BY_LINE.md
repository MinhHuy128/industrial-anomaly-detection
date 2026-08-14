# 💻 CHUYÊN ĐỀ 3: MAPPING CODE CHI TIẾT TỪNG DÒNG (LINE-BY-LINE CODE EXPLANATION)

---

## 1. FILE `src/models/vitill_gct.py` (MÔ HÌNH CHÍNH)

### A. Hàm Nạp Backbone DINOv2-Register (`load_dinov2_register`)
```python
def load_dinov2_register(device: torch.device) -> nn.Module:
    hub_dir = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    if hub_dir.exists():
        backbone = torch.hub.load(str(hub_dir), 'dinov2_vitb14_reg', source='local').to(device)
```
- **Chức năng:** Nạp mô hình DINOv2-Register ViT-B/14.
- **Tại sao có `source='local'`?** Nếu thư mục cache đã tồn tại trên đĩa cứng, PyTorch Hub sẽ đọc thẳng từ đĩa mà **KHÔNG CẦN KẾT NỐI INTERNET / KHÔNG TẢI LẠI** (Nạp tức thì trong 0.1 giây!).
- **Input:** `device` (`torch.device('cuda')` hoặc `'cpu'`).
- **Output:** Module PyTorch đã được đóng băng (`requires_grad = False`).

---

### B. Lớp `ViTillGCT.__init__` & `forward`
```python
class ViTillGCT(nn.Module):
    def __init__(self, embed_dim=768, num_decoder_layers=8, target_layers=[2,3,4,5,6,7,8,9], gct_lambda=0.5):
        super().__init__()
        self.gct_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.gct_proj  = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
```
- **Chức năng:** Khởi tạo GCT Token học được và Projection Head 1 lớp.
- **Tại sao nhân `* 0.02`?** Đảm bảo khởi tạo chuẩn Gaussian $\mathcal{N}(0, 0.02^2)$ giúp tham số không bị bộc phát giá trị quá lớn lúc mới bắt đầu train.
- **Kích thước Tensor:** `gct_token`: $[1, 1, 768]$.

```python
    def forward(self, en_list, cls_token=None):
        # 1. Layer fusion
        x = sum(en_list) / len(en_list)          # [B, 784, 768]
        x = self.bottleneck(x)                   # [B, 784, 768]

        # 2. Prepend GCT token
        gct_tok = self.gct_token.expand(x.shape[0], -1, -1) # [B, 1, 768]
        x_in    = torch.cat([gct_tok, x], dim=1)           # [B, 785, 768]

        # 3. Pass through 8 Decoder blocks
        x_dec   = self.decoder(x_in)                       # [B, 785, 768]
        
        # 4. Separate decoded patch tokens & GCT token
        gct_out   = x_dec[:, 0:1, :]                       # [B, 1, 768]
        de_patches = x_dec[:, 1:, :]                       # [B, 784, 768]

        # 5. Project GCT token
        proj_gct  = self.gct_proj(gct_out.squeeze(1))      # [B, 768]

        # 6. Compute GCT loss if CLS token provided
        if cls_token is not None:
            gct_loss = compute_gct_loss(proj_gct, cls_token.detach())
            return [x], [de_patches], gct_loss
        return [x], [de_patches]
```
- **Kích thước Ma trận theo từng bước:**
  - `en_list`: Danh sách 8 Tensor $[B, 784, 768]$.
  - `x`: Cộng trung bình $\rightarrow [B, 784, 768]$.
  - `gct_tok`: Mở rộng theo Batch size $B \rightarrow [B, 1, 768]$.
  - `x_in`: Ghép nối theo chiều chiều dài chuỗi `dim=1` $\rightarrow [B, 785, 768]$.
  - `gct_out`: Cắt phần tử đầu tiên tại chỉ số `0:1` $\rightarrow [B, 1, 768]$.
  - `de_patches`: Cắt các phần tử còn lại từ chỉ số `1:` $\rightarrow [B, 784, 768]$.
  - `cls_token.detach()`: **QUAN TRỌNG!** Đứt mạch gradient của CLS token từ backbone.

---

## 2. FILE `src/eval.py` (HÀM ĐÁNH GIÁ & QUY ĐỔI TỌA ĐỘ SPATIAL ALIGNMENT)

### A. Hàm Căn Chỉnh Tọa Độ Không Gian (`compute_anomaly_map`)
```python
def compute_anomaly_map(en_list, de_list, crop_size: int = 392, out_size: int = 448) -> np.ndarray:
    anomaly_map = torch.zeros(1, 1, 1, 1, device=en_list[0].device)
    for en, de in zip(en_list, de_list):
        a_map = 1.0 - F.cosine_similarity(en, de, dim=1, eps=1e-8)  # [B, 28, 28]
        a_map = a_map.unsqueeze(1)                                    # [B, 1, 28, 28]
        a_map = F.interpolate(a_map, size=(crop_size, crop_size), mode='bilinear') # [B, 1, 392, 392]
        anomaly_map = anomaly_map + a_map

    crop_amap = (anomaly_map / len(en_list)).squeeze().cpu().numpy() # [392, 392]

    # Paste into 448x448 canvas
    canvas = np.zeros((out_size, out_size), dtype=crop_amap.dtype)
    top  = (out_size - crop_size) // 2  # (448 - 392) // 2 = 28
    left = (out_size - crop_size) // 2  # 28
    canvas[top:top+crop_size, left:left+crop_size] = crop_amap  # canvas[28:420, 28:420]
    return gaussian_filter(canvas, sigma=4)
```
- **Chức năng:** Tính bản đồ bất thường theo từng patch, upsample lên $392 \times 392$, sau đó dán vào giữa canvas $448 \times 448$ tại offset `top=28, left=28`.
- **Tại sao phải làm như vậy?**
  - Ảnh gốc đầu vào có kích thước $448 \times 448$, sau đó bị `CenterCrop(392)` lấy vùng trung tâm $392 \times 392$.
  - Nếu không dán lại vào canvas $448 \times 448$, bản đồ bất thường sẽ bị lệch tọa độ so với mặt nạ Ground Truth Mask $\rightarrow$ sPRO bị tụt dốc thảm hại.
  - Việc dán lại giúp khôi phục tọa độ chuẩn xác 100%, nâng sPRO từ `60.90%` lên **`70.10%`**!

---

### B. Hàm Tính Điểm Ảnh (`image_score`)
```python
def image_score(anomaly_map: np.ndarray, max_ratio: float = 0.01) -> float:
    flat = anomaly_map.flatten()
    k    = max(1, int(len(flat) * max_ratio))
    return float(np.sort(flat)[-k:].mean())
```
- **Chức năng:** Trải phẳng bản đồ bất thường và lấy trung bình của 1% số pixel có điểm cao nhất (`top-1% mean percentile pooling`).
- **Tác dụng:** Loại bỏ nhiễu cực đoan của 1 pixel đơn lẻ, giúp điểm ảnh $\text{Score}_{\text{patch}}$ phản ánh đúng mức độ bất thường vùng.

---

### C. Công Thức Dual-Stream Active Scoring (`infer_one`)
```python
amap = compute_anomaly_map(en, de, crop_size=crop_size, out_size=out_size)
score_patch = image_score(amap)
score_final = score_patch + gamma * score_gct  # gamma = 1.0 (Fixed Coefficient)
```
- **Chức năng:** Kết hợp điểm Patch cục bộ với điểm GCT đại cục theo trọng số cố định $\gamma = 1.0$.
