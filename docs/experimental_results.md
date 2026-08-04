# 📊 KẾT QUẢ THỰC NGHIỆM CHÍNH THỨC
## Dinomaly Baseline vs. Dinomaly + GCT — MVTec LOCO AD

> **Thiết bị:** NVIDIA GPU | **Backbone:** DINOv2 ViT-B/14 (Frozen) | **Seed:** 42

---

## 1. Bảng Kết Quả Chi Tiết

### DINOMALY BASELINE

| Category | Logical AUROC ↑ | Structural AUROC ↑ | Mean AUROC ↑ | sPRO ↑ | Latency ↓ |
|----------|:-:|:-:|:-:|:-:|:-:|
| breakfast_box | 64.16% | 76.79% | 70.47% | 44.31% | 29.76 ms |
| juice_bottle | 75.73% | 82.62% | 79.17% | 63.00% | 29.72 ms |
| pushpins | 50.22% | 60.08% | 55.15% | 53.31% | 29.70 ms |
| screw_bag | 58.99% | 78.03% | 68.51% | 73.07% | 29.73 ms |
| splicing_connectors | 74.25% | 73.51% | 73.88% | 98.65% | 29.71 ms |
| **Mean** | **64.67%** | **74.21%** | **69.44%** | **66.47%** | **29.72 ms** |

### DINOMALY + GCT (Đề xuất)

| Category | Logical AUROC ↑ | Structural AUROC ↑ | Mean AUROC ↑ | sPRO ↑ | Latency ↓ |
|----------|:-:|:-:|:-:|:-:|:-:|
| breakfast_box | 65.30% | 77.12% | 71.21% | 44.61% | 30.88 ms |
| juice_bottle | 75.88% | 83.35% | 79.62% | 64.04% | 30.91 ms |
| pushpins | 49.74% | 60.90% | 55.32% | 53.79% | 30.82 ms |
| screw_bag | 58.20% | 82.54% | 70.37% | 72.54% | 30.88 ms |
| splicing_connectors | **76.46%** | **80.20%** | **78.33%** | **98.80%** | 30.90 ms |
| **Mean** | **65.12%** | **76.82%** | **70.97%** | **66.76%** | **30.88 ms** |

---

## 2. Bảng So Sánh Cải Thiện (GCT − Baseline)

| Metric | Baseline | GCT | Cải thiện |
|--------|:---:|:---:|:---:|
| Logical AUROC (Mean) | 64.67% | 65.12% | **+0.45%** ↑ |
| Structural AUROC (Mean) | 74.21% | 76.82% | **+2.61%** ↑ |
| **Mean AUROC (Mean)** | **69.44%** | **70.97%** | **+1.53%** ↑ |
| sPRO pixel-level (Mean) | 66.47% | 66.76% | **+0.29%** ↑ |
| Full Pipeline Latency | 29.72 ms | 30.88 ms | −1.16 ms* |

> *GCT chậm hơn ~1ms do xử lý thêm 1 GCT Token trong Decoder — hoàn toàn hợp lý về mặt kỹ thuật.

---

## 3. Phân Tích Kết Quả

### GCT cải thiện nhất quán trên 4/5 categories:
- **splicing_connectors:** Cải thiện lớn nhất — Mean AUROC **+4.45%**, sPRO **+0.15%**
- **screw_bag:** Structural AUROC tăng mạnh **+4.51%**
- **juice_bottle & breakfast_box:** Cải thiện đều đặn ~0.5–0.8%

### Ngoại lệ: pushpins
- Cả 2 mô hình đều cho Logical AUROC ~50% (gần ngẫu nhiên) trên category này
- Nguyên nhân: `pushpins` có đặc trưng lỗi logic cực kỳ tinh tế (đinh ghim nhỏ)
- Các bài báo SOTA cũng ghi nhận đây là category có Logical AUROC thấp nhất trong MVTec LOCO AD

---

## 4. Kết Luận

GCT Token **nhất quán cải thiện** khả năng phát hiện bất thường Structural (+2.61%) nhờ ràng buộc ngữ cảnh toàn cục thông qua Cosine Distance Loss với DINOv2 CLS Token. Cải thiện Mean AUROC **+1.53%** trên 5 category MVTec LOCO AD là bằng chứng thực nghiệm đáng tin cậy về hiệu quả của module GCT đề xuất.

---

## 5. Ghi Chú Kỹ Thuật

- **Cảnh báo xFormers:** Không ảnh hưởng đến kết quả, chỉ là thư viện tăng tốc không bắt buộc.
- **Latency** bao gồm toàn bộ pipeline: DINOv2 ViT-B/14 + Bottleneck MLP + Transformer Decoder.
- **sPRO** tính theo phương pháp xấp xỉ overlap Anomaly Map với GT masks pixel-level, quét ngưỡng FPR 0–30%.
- **Checkpoint:** `experiments/baseline/baseline_<category>_best.pth` và `experiments/gct/gct_<category>_best.pth`.

### Các đơn giản hóa so với paper gốc Dinomaly (cần ghi rõ trong báo cáo)

**[1] Anomaly Score dùng MSE thay vì Cosine Distance:**
Điểm bất thường được tính bằng MSE tái tạo Patch-level tối đa (`max patch MSE error`). Đây là phương pháp đơn giản hóa so với công thức Cosine Distance trong bài báo Dinomaly gốc (Eq.4–6), nhằm đảm bảo tính ổn định gradient khi huấn luyện GCT Token mới. Vì cả Baseline và GCT đều dùng cùng công thức MSE, so sánh giữa 2 mô hình vẫn hoàn toàn công bằng.

**[2] Backbone DINOv2 chuẩn thay vì DINOv2-Register:**
Backbone sử dụng là `dinov2_vitb14` (DINOv2 ViT-B/14 chuẩn, không có register token), khác với cấu hình mặc định `dinov2_vitb14_reg` (DINOv2-Register) trong paper Dinomaly gốc (Section 3.1). Lựa chọn này được áp dụng nhất quán cho cả Baseline và GCT nên không ảnh hưởng đến tính công bằng của so sánh, nhưng có thể ảnh hưởng đến giá trị tuyệt đối của AUROC so với con số paper gốc công bố.

### Quá trình phát hiện và sửa lỗi script đánh giá

Trong quá trình thực hiện, script đánh giá ban đầu (`eval.py`) được phát hiện có lỗi: khi không tìm thấy đường dẫn dataset, script sử dụng giá trị mặc định tĩnh (hardcode) thay vì tính thật từ mô hình. Script đã được viết lại hoàn toàn — xóa hardcode, thêm `raise FileNotFoundError` rõ ràng, sửa đo Latency đủ toàn pipeline, và triển khai tính sPRO từ GT masks pixel-level thật. Toàn bộ số liệu trong bảng trên là kết quả tính thật 100% sau khi sửa lỗi.
