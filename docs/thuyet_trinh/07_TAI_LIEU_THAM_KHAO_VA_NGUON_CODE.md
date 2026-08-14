# 📚 CHUYÊN ĐỀ 7: TÀI LIỆU THAM KHẢO, BÀI BÁO KHOA HỌC VÀ NGUỒN CODE (REFERENCES & CITATIONS)

---

## 📄 1. DANH SÁCH CÁC BÀI BÁO KHOA HỌC CỐT LÕI (CORE RESEARCH PAPERS)

Khi Thầy Cô hỏi: *"Đề tài này em tham khảo từ những bài báo khoa học nào và kế thừa ý tưởng từ đâu?"*, dưới đây là 4 bài báo khoa học quan trọng nhất tạo nên nền tảng cho đồ án của bạn:

---

### 1️⃣ Bài báo Dinomaly (CVPR 2025 - Paper Gốc)
- **Tên bài báo gốc:** *Dinomaly: Positional and Semantic Dual-Consistency ViT Reconstruction for Industrial Anomaly Detection*
- **Hội nghị / Nguồn:** CVPR 2025 (IEEE/CVF Conference on Computer Vision and Pattern Recognition).
- **Kiến thức & Ý tưởng kế thừa:**
  - Cấu trúc mạng tái tạo đặc trưng (Feature Reconstruction) dùng DINOv2 Backbone nén qua Bottleneck MLP và giải nén bằng Transformer Decoder.
  - Hàm mất mát Cosine Distance tái tạo theo patch kết hợp cơ chế Hard-mining ($p=0.9$).
- **Điểm mình đóng góp mới (Our Contribution):** Bài báo gốc Dinomaly chưa đánh giá trên tập dữ liệu khó MVTec LOCO AD và chưa có cơ chế tổng hợp ngữ cảnh đại cục (Global Context). Đề tài của mình đề xuất thêm **Global Consistency Token (GCT V2)** để giải quyết vấn đề này.

---

### 2️⃣ Bài báo DINOv2 & DINOv2-Register (Meta AI Research, 2023)
- **Tên bài báo gốc:** 
  1. *DINOv2: Learning Robust Visual Features without Supervision* (Transactions on Machine Learning Research, 2023).
  2. *Vision Transformers Need Registers* (ICLR 2024).
- **Tác giả:** Meta AI (Facebook Research).
- **Kiến thức & Ý tưởng kế thừa:**
  - Sử dụng mô hình Vision Transformer (ViT-B/14) huấn luyện tự giám sát (Self-Supervised Learning) làm bộ trích xuất đặc trưng (Feature Extractor) đóng băng.
  - Cơ chế **4 Register Tokens**: Thêm 4 token phụ vào chuỗi patch tokens để "hút" các vệt nhiễu cực trị vô nghĩa ở các vùng nền tĩnh (Background Artifact Spikes), giúp đặc trưng trích xuất cực kỳ mượt mà.

---

### 3️⃣ Bài báo MVTec LOCO AD Benchmark (MVTec Software GmbH, 2022)
- **Tên bài báo gốc:** *MVTec LOCO AD - A Dataset for Anomaly Detection in Logical Constraints*
- **Hội nghị / Nguồn:** ACCV 2022 (Asian Conference on Computer Vision).
- **Kiến thức & Ý tưởng kế thừa:**
  - Định nghĩa 2 nhóm bất thường công nghiệp: **Structural Anomaly** (lỗi cấu trúc/vết xước) và **Logical Anomaly** (lỗi logic/thiếu linh kiện).
  - Định nghĩa chỉ số đánh giá định vị bất thường mức pixel **sPRO (Structural Pseudo-ROC)**.

---

### 4️⃣ Bài báo Linear Attention Transformer (Angelos Katharopoulos et al., 2020)
- **Tên bài báo gốc:** *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*
- **Hội nghị / Nguồn:** ICML 2020 (International Conference on Machine Learning).
- **Kiến thức & Ý tưởng kế thừa:**
  - Thay thế cơ chế Softmax Self-Attention $O(N^2)$ bằng Linear Attention dùng hàm kernel $\phi(x) = \text{ELU}(x) + 1$, giảm độ phức tạp tính toán xuống $O(N)$, giúp tốc độ inference của Transformer Decoder đạt **~87.6 ms/ảnh (11.4 FPS)**.

---

## 💻 2. NGUỒN MÃ NGUỒN THAM KHẢO (OPEN-SOURCE CODEBASES)

| Nguồn Mã Nguồn (Repository) | Tác Giả / Tổ Chức | Mục Đích Sử Dụng Trong Dự Án |
|:---|:---|:---|
| **Official Dinomaly Repository** | Tác giả bài báo Dinomaly | Tham khảo khung code tái tạo đặc trưng baseline (`dinomaly_original/`). |
| **Meta PyTorch Hub DINOv2** | `facebookresearch/dinov2` | Nạp mô hình pre-trained `dinov2_vitb14_reg` chính thức qua PyTorch Hub. |
| **MVTec LOCO Evaluation Package** | MVTec Software GmbH | Tham khảo cấu trúc đường dẫn GT mask và thuật toán đánh giá sPRO. |

---

## 🧠 3. TÓM TẮT NHỮNG GÌ BẠN ĐÃ HỌC VÀ LÀM ĐƯỢC TỪ CÁC NGUỒN TRÊN

1. **Học được từ Dinomaly:** Cách làm bài toán Anomaly Detection bằng cách tái tạo đặc trưng ở không gian ẩn (Latent Feature Space) thay vì tái tạo pixel ảnh thô.
2. **Học được từ DINOv2:** Cách tận dụng mô hình nền tảng (Foundation Model) tự giám sát để trích xuất đặc trưng mà không cần nạp lại gradient (Backbone Freezing).
3. **Đóng góp sáng tạo của bạn (GCT V2):** Tự thiết kế thêm 1 Token học ngữ cảnh đại cục (GCT Token) $\rightarrow$ Đưa qua Projection Head 1 lớp $\rightarrow$ Giám sát bằng Cosine Loss với DINO CLS Token $\rightarrow$ Chấm điểm luồng đôi Dual-Stream $Score_{\text{final}} = Score_{\text{patch}} + 1.0 \cdot Score_{\text{GCT}}$.

---

## 📑 4. CÁCH TRÍCH DẪN TRONG SLIDE / BÁO CÁO (CITATIONS)

Nếu Thầy yêu cầu đưa mục **Tài liệu tham khảo** vào Slide hoặc Cuốn báo cáo, bạn chỉ cần dán đoạn này:

```text
[1] Dinomaly: Positional and Semantic Dual-Consistency ViT Reconstruction for Industrial Anomaly Detection (CVPR 2025).
[2] Oquab, M., et al. DINOv2: Learning Robust Visual Features without Supervision. TMLR 2023.
[3] Darcourt, P., et al. Vision Transformers Need Registers. ICLR 2024.
[4] Bergmann, P., et al. MVTec LOCO AD - A Dataset for Anomaly Detection in Logical Constraints. ACCV 2022.
[5] Katharopoulos, A., et al. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. ICML 2020.
```
