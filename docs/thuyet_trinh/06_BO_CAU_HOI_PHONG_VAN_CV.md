# 💼 CHUYÊN ĐỀ 6: BỘ CÂU HỎI PHỎNG VẤN TUYỂN DỤNG AI / COMPUTER VISION ENGINEER (CV INTERVIEW PREPARATION)

---

## 📄 1. CÁCH VIẾT DỰ ÁN VÀO CV (RESUME BULLET POINTS)

Hãy đưa dự án này vào mục **Projects** trong CV của bạn với các dòng mô tả ấn tượng (Impact-driven):

```text
INDUSTRIAL ANOMALY DETECTION ON MVTEC LOCO AD WITH GLOBAL CONSISTENCY TOKEN (GCT)
Role: Lead AI/CV Engineer | Tech Stack: PyTorch, DINOv2-Register, ViT, Linear Attention, CUDA

• Designed and implemented ViTill-GCT, an active dual-stream anomaly detection architecture leveraging DINOv2-Register ViT-B/14 backbone and a learned Global Consistency Token (GCT).
• Boosted Logical Anomaly AUROC on MVTec LOCO AD benchmark from 76.40% to 80.33% (+3.93% gain over baseline), setting a record +9.44% boost on the complex 'screw_bag' category.
• Developed an active dual-stream scoring mechanism (Score_final = Score_patch + 1.0 * Score_gct) with zero test-set tuning bias (a-priori fixed gamma=1.0).
• Optimized spatial coordinate restoration (392 -> 448 CenterCrop padding), maintaining a pixel-level sPRO localization score of 70.10% at real-time industrial speed (~11.4 FPS / 87.6ms on GPU).
• Implemented robust PyTorch optimizations: gradient hook auto-cleanup to prevent VRAM memory leaks, offline local hub caching for zero-latency backbone loading, and linear warmup for hard-mining loss.
```

---

## 🎙️ 2. CÂU HỎI MỞ ĐẦU: "EM HÃY GIỚI THIỆU VỀ DỰ ÁN NÀY TRONG 2 PHÚT" (ELEVATOR PITCH)

### 💬 Trả lời mẫu:
> *"Em chào Anh/Chị. Dự án ấn tượng nhất của em là nghiên cứu và phát hiện bất thường công nghiệp trên tập dữ liệu MVTec LOCO AD.*
>
> *Thách thức lớn nhất của bài toán này là các lỗi logic (Logical Anomalies) — ví dụ như thiếu linh kiện hay sai vị trí. Các mô hình tái tạo cục bộ thông thường (Baseline) thường bỏ sót các lỗi này vì từng ô vuông nhỏ đều trông rất hoàn hảo.*
>
> *Để giải quyết, em đã đề xuất kiến trúc **ViTill-GCT V2** tích hợp một **Global Consistency Token (GCT)** vào Transformer Decoder. Token này đóng vai trò học ngữ cảnh đại cục toàn ảnh và được giám sát bởi vector CLS của DINOv2-Register.*
>
> *Kết quả là mô hình của em đã nâng chỉ số **Logical AUROC từ 76.40% lên 80.33% (+3.93% so với baseline)**, đặc biệt tăng vọt **+9.44%** trên sản phẩm `screw_bag`. Tốc độ suy luận đạt **~11.4 FPS (87.6ms/ảnh)** trên GPU, hoàn toàn đáp ứng chuẩn thời gian thực cho nhà máy ạ!"*

---

## 🎯 3. BỘ 15 CÂU HỎI PHỎNG VẤN CHUYÊN SÂU (TECHNICAL DEEP-DIVE INTERVIEW QUESTIONS)

### ❓ Q1: "Tại sao em lại chọn DINOv2 thay vì ResNet hay EfficientNet làm Backbone?"
- **Trả lời:** *"DINOv2 được huấn luyện tự giám sát (Self-Supervised) trên tập dữ liệu khổng lồ 142 triệu ảnh. Không gian đặc trưng của DINOv2 chứa thông tin ngữ nghĩa và tương quan không gian mạnh mẽ hơn rất nhiều so với ResNet (chỉ được train supervised trên ImageNet). Đặc biệt, bản DINOv2 ViT-B/14 cho ra 784 spatial patch tokens rất phù hợp cho bài toán tái tạo đặc trưng không gian."*

---

### ❓ Q2: "Sự khác biệt giữa GCT V1 (Passive) và GCT V2 (Active) là gì?"
- **Trả lời:** *"GCT V1 chỉ dùng token trong quá trình huấn luyện (Passive), lúc suy luận (Inference) chỉ dùng điểm tái tạo patch. V1 chỉ đạt 84.64% AUROC (bằng baseline). GCT V2 chuyển sang cơ chế Active Dual-Stream Scoring: kết hợp trực tiếp điểm $Score_{GCT}$ đại cục vào công thức chấm điểm lúc inference ($Score_{patch} + 1.0 \cdot Score_{GCT}$), giúp kích hoạt còi báo động ngay khi có bất thường logic. Nhờ đó V2 bứt phá lên 86.68% Mean AUROC."*

---

### ❓ Q3: "Tại sao em không dùng Softmax Attention mà dùng Linear Attention trong Decoder?"
- **Trả lời:** *"Softmax Attention có độ phức tạp $O(N^2)$. Với $N=784$ patch tokens, phép nhân $784 \times 784$ tiêu tốn rất nhiều bộ nhớ và thời gian. Linear Attention dùng hàm kích hoạt ELU+1 để tách ma trận Kernel $\phi(Q)\phi(K)^T V$, đưa độ phức tạp về $O(N)$, giúp tốc độ inference đạt 87.6ms/ảnh."*

---

### ❓ Q4: "Em đã giải quyết hiện tượng Gradient Sinking ở Projection Head như thế nào?"
- **Trả lời:** *"Ban đầu nếu dùng Projection Head 2 lớp MLP phức tạp, các lớp tuyến tính có xu hướng 'hấp thụ' hết gradient của GCT Loss, khiến tín hiệu không truyền ngược được về các lớp Transformer Decoder. Em đã tối giản Projection Head xuống 1 lớp `Linear(768, 768) + LayerNorm`, giúp gradient trôi mượt mà về Decoder."*

---

### ❓ Q5: "Tại sao em lại dùng Hard-mining Cosine Loss với $p=0.9$?"
- **Trả lời:** *"Trong một bức ảnh sản phẩm, đa số các vùng nền (Background) đều rất dễ tái tạo. Nếu lấy trung bình toàn bộ 784 patch, tín hiệu lỗi ở vùng bất thường nhỏ sẽ bị pha loãng. Hệ số $p=0.9$ giúp lọc ra Top 10% các patch có khoảng cách cosine lớn nhất (khó tái tạo nhất) để tính loss, tập trung lực gradient vào đúng vùng bị lỗi."*

---

### ❓ Q6: "Em đã làm gì để tránh Data Leakage / Test-Set Overfitting khi chọn tham số $\gamma$?"
- **Trả lời:** *"Em không dùng phương pháp sweep tìm $\gamma$ tối ưu trên tập test (Oracle Sweep). Em cố định $\gamma=1.0$ (Fixed Coefficient Weighting) a-priori trước khi chạy tập test. Mặc dù Oracle sweep có thể đạt 87.27%, em chọn con số 86.68% ($\gamma=1.0$) để đảm bảo 100% tính nguyên tắc khoa học và khả năng tổng quát hóa trên dữ liệu thực tế."*

---

### ❓ Q7: "Làm thế nào em phát hiện và fix lỗi VRAM Memory Leak trong PyTorch?"
- **Trả lời:** *"Em theo dõi dung lượng VRAM bằng `torch.cuda.memory_allocated()` qua từng iteration. Em phát hiện VRAM tăng tuyến tính do Gradient Hooks trong custom loss không được giải phóng. Em đã dùng `handle.remove()` để tự hủy hook ngay sau bước `backward()`, giữ VRAM phẳng ổn định suốt 5000 iterations."*

---

### ❓ Q8: "Em làm sao để deploy mô hình này lên môi trường Production thực tế?"
- **Trả lời:** *"Để deploy lên edge device hoặc camera nhà máy, em sẽ export mô hình PyTorch sang TensorRT hoặc ONNX Runtime, thực hiện Quantization INT8/FP16 cho Decoder. Do DINOv2 backbone đã đóng băng, ta có thể pre-cache hoặc dùng TensorRT engine tối ưu hóa cực mạnh, đưa latency xuống dưới 20ms/ảnh."*

---

### ❓ Q9: "Nếu tập dữ liệu train bị lẫn 5% ảnh lỗi (Noisy Training Data), mô hình em có chạy được không?"
- **Trả lời:** *"Mô hình của em có cơ chế Hard-mining Warmup ($p$ tăng từ $0 \rightarrow 0.9$ trong 1000 iters đầu) giúp mô hình học đặc trưng nền ổn định trước. Tuy nhiên nếu ảnh lỗi xuất hiện nhiều trong train set, Decoder sẽ học cách tái tạo luôn cả lỗi. Để khắc phục, ta có thể áp dụng thêm kỹ thuật Memory Bank hoặc Synthetic Anomaly Augmentation (CutPaste/DRAEM) để tăng tính bền vững."*

---

### ❓ Q10: "Em đo latency như thế nào cho chính xác?"
- **Trả lời:** *"Em dùng phương pháp Single-image Repeated Inference: khởi động GPU với 10 warmup runs (để trút hết overhead nạp CUDA kernel), sau đó đo thời gian trung bình qua 50 lần chạy lặp lại liên tiếp với `batch_size=1` bằng `torch.cuda.Event(enable_timing=True)` để đảm bảo độ chính xác ms trên GPU."*

---

### ❓ Q11: "Tại sao sPRO lại quan trọng hơn Pixel AUROC trong bài toán Anomaly Localization?"
- **Trả lời:** *"Pixel AUROC có nhược điểm bị chi phối bởi các vùng background lớn. sPRO (Structural Pseudo-ROC) đánh giá độ đè phủ (Overlap Ratio) trên từng vùng liên thông (Connected Component) của vết lỗi. Ngay cả khi vết lỗi rất nhỏ, sPRO vẫn bắt buộc mô hình phải đè phủ tối thiểu ngưỡng $\text{PRO}$ trên từng vùng lỗi riêng biệt."*

---

### ❓ Q12: "Em đã bao giờ gặp bug `NaN Loss` chưa và sửa thế nào?"
- **Trả lời:** *"Dạ có, khi tính Cosine Distance $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$, nếu một vector có norm bằng 0 thì phép chia sẽ ra `NaN`. Em đã thêm hằng số bù `eps=1e-8` trong `F.cosine_similarity` và dùng `torch.nan_to_num()` để triệt tiêu hoàn toàn lỗi NaN."*

---

### ❓ Q13: "Nếu muốn cải tiến dự án này xa hơn nữa (Future Work), em sẽ làm gì?"
- **Trả lời:** *"Em sẽ nghiên cứu 2 hướng: (1) Thay GCT 1 token bằng Multi-scale Context Tokens để captured ngữ cảnh ở nhiều độ phân giải khác nhau; (2) Tích hợp cơ chế Contrastive Learning giữa GCT Token của ảnh Good và các mẫu bất thường tổng hợp (Synthetic Anomalies) để tăng biên độ phân tách bộc phát của $Score_{GCT}$."*

---

### ❓ Q14: "DINOv2 ViT-B/14 có 86M params, liệu nó có quá nặng cho nhà máy không?"
- **Trả lời:** *"DINOv2 Backbone được đóng băng hoàn toàn (`requires_grad=False`), không tham gia vào quá trình tính gradient lúc train. Khi inference, ta có thể dùng kỹ thuật Knowledge Distillation để chắt lọc kiến thức từ DINOv2 sang một mô hình nhỏ gọn như ViT-Small (22M params) hoặc MobileNetV4 để chạy trên thiết bị nhúng giá rẻ."*

---

### ❓ Q15: "Bài học lớn nhất em rút ra được từ dự án này là gì?"
- **Trả lời:** *"Bài học lớn nhất của em là tư duy Chẩn đoán Kỹ thuật dựa trên Thực nghiệm (Empirical Diagnosis). Đứng trước một mô hình không chạy như ý, không được đoán mò mà phải đo đạc phân phối score, kiểm tra gradient flow, soi từng ma trận tensor shape và tuân thủ chặt chẽ nguyên tắc khách quan khoa học."*
