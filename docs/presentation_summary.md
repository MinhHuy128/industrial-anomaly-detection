# 🎯 TỔNG HỢP NỘI DUNG BÁO CÁO ĐỒ ÁN (PRESENTATION SUMMARY)
## Đề Tài: Phát Hiện Bất Thường Công Nghiệp Trên MVTec LOCO AD Với GCT (Global Consistency Token)

---

## 1. MỤC TIÊU NGHIÊN CỨU & ĐÓNG GÓP CHÍNH
- **Bài toán:** Phát hiện bất thường cấu trúc (Structural) và bất thường logic (Logical Anomalies - mất linh kiện, sai vị trí) trên tập dữ liệu công nghiệp MVTec LOCO AD.
- **Giải pháp đề xuất:** Tích hợp module **Global Consistency Token (GCT V2 - Active Dual-Stream)** vào kiến trúc Dinomaly để tổng hợp ngữ cảnh đại cục (Global Context), khắc phục điểm yếu của luồng tái tạo Patch cục bộ.

---

## 2. BẢNG KẾT QUẢ THỰC NGHIỆM CHÍNH THỨC (BENCHMARK RESULTS)

> **Thiết lập:** Trọng số hệ số cố định $\gamma = 1.0$ cho GCT Score ($\text{Score}_{\text{final}} = \text{Score}_{\text{patch}} + 1.0 \cdot \text{Score}_{\text{GCT}}$), chạy trên 5 tập sản phẩm công nghiệp.

| Hạng Mục Sản Phẩm (Category) | Baseline (Dinomaly Paper) | Mô Hình Đề Xuất (GCT V2) | Mức Mức Tăng (Gain) |
|:---|:---:|:---:|:---:|
| **Breakfast Box** (Logical / Mean AUROC) | 88.97% / 90.74% | **91.94% / 91.32%** | **+2.97% / +0.58%** |
| **Juice Bottle** (Logical / Mean AUROC) | 90.74% / 94.47% | **94.10% / 96.02%** | **+3.36% / +1.55%** |
| **Pushpins** (Logical / Mean AUROC) | 54.90% / 68.07% | **56.64% / 69.82%** | **+1.74% / +1.75%** |
| **Screw Bag** (Logical / Mean AUROC) | 59.19% / 76.21% | **68.63% / 81.44%** | 🔥 **+9.44% / +5.23%** |
| **Splicing Connectors** (Logical / Mean AUROC) | 88.19% / 93.85% | **90.32% / 94.81%** | **+2.13% / +0.96%** |
| **TRUNG BÌNH (5 CATEGORIES)** | | | |
| • **Logical Anomaly AUROC** | **76.40%** | **80.33%** | 🏆 **+3.93% (Bứt phá mạnh)** |
| • **Structural Anomaly AUROC** | **92.97%** | **93.04%** | **+0.07% (Duy trì hiệu năng)** |
| • **MEAN AUROC SCORE** | **84.67%** | **86.68%** | 🎯 **+2.01% (Tốt nhất đề tài)** |
| • **Độ Trễ Inference (Latency)** | **86.97 ms / ảnh** | **87.60 ms / ảnh** | **~11.4 FPS (Đáp ứng thời gian thực)** |

---

## 3. 3 LUẬN ĐIỂM KỸ THUẬT QUAN TRỌNG KHI BẢO VỆ TRƯỚC HỘI ĐỒNG

1. **Tại sao Logical AUROC tăng mạnh (+3.93%) nhưng Structural AUROC giữ nguyên?**
   - Lỗi cấu trúc (Structural) chỉ xuất hiện ở các vết xước/móp cục bộ, luồng Patch tái tạo đã làm rất tốt. 
   - Lỗi logic (Logical - thiếu ốc/vít, sai thứ tự) đòi hỏi hiểu biết đại cục. GCT Token đóng vai trò tổng hợp ngữ cảnh toàn ảnh, giúp nhận diện sự mất cân bằng ngữ cảnh mà luồng Patch cục bộ bị bỏ sót.

2. **Ý nghĩa của hệ số cố định $\gamma = 1.0$ (Fixed Coefficient Weighting):**
   - Giá trị $\gamma = 1.0$ được thiết lập cố định a-priori (trước khi đánh giá tập test) để tránh hiện tượng *test-set tuning bias*.
   - Phân tích phân phối cho thấy $\text{Score}_{\text{GCT}}$ bình thường rất nhỏ ($\approx 0.002$), nhưng khi gặp lỗi logic sẽ bộc phát mức tăng vọt (đạt đỉnh tới $0.1434$), kích hoạt luồng cảnh báo tức thì.

3. **Căn chỉnh tọa độ không gian (Spatial Coordinate Restoration $392 \rightarrow 448$):**
   - Ảnh đầu vào được resize $448 \times 448$ và CenterCrop $392 \times 392$. Bản đồ bất thường đầu ra $28 \times 28$ được upsample về $392 \times 392$ rồi dán lại vào canvas $448 \times 448$ (offset `top=28, left=28`) để khớp chính xác với hệ tọa độ mặt nạ Ground Truth.

---

## 4. TÀI LIỆU VÀ CÁC THƯ MỤC TRỰC QUAN
- **Ảnh Heatmap trực quan:** `docs/figures/heatmap_*.png`
- **Biểu đồ so sánh AUROC:** `docs/figures/bar_chart_logical_auroc.png`
- **Script kiểm tra nhanh:** `python src/eval.py --category breakfast_box --use_gct`
