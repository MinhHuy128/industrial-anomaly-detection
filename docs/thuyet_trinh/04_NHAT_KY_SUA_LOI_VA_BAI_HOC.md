# 🛠️ CHUYÊN ĐỀ 4: NHẬT KÝ SỬA LỖI BUG THỰC TẾ VÀ BÀI HỌC KINH NGHIỆM (BUG LOG & LESSONS LEARNED)

---

## 📌 TỔNG HỢP 7 LỖI BUG THỰC TẾ ĐÃ PHÁT HIỆN VÀ KHẮC PHỤC

Khi Thầy hoặc Nhà tuyển dụng hỏi: *"Trong quá trình thực hiện đồ án, em đã gặp những khó khăn hay bug kỹ thuật nào lớn nhất và em đã giải quyết nó ra sao?"*, dưới đây là 7 câu chuyện thực tế đắt giá nhất bạn có thể tự tin chia sẻ:

---

### 1️⃣ BUG 1: Gradient Leakage (Trôi Gradient hỏng Backbone)
- **Triệu chứng:** Khi huấn luyện GCT Loss, điểm Mean AUROC của mô hình bị suy giảm thảm hại sau mỗi epoch, đặc trưng của DINOv2 bị biến dạng.
- **Nguyên nhân root-cause:** Do khi truyền vector `cls_token` từ DINOv2 qua hàm tính `compute_gct_loss`, em đã quên ngắt mạch gradient (`.detach()`). Kết quả là gradient từ GCT Loss trôi ngược về thay đổi trọng số của DINOv2 Backbone (vốn bắt buộc phải đóng băng `requires_grad=False`).
- **Cách khắc phục:** 
  Thêm `.detach()` vào CLS token trước khi đưa vào hàm Loss:
  ```python
  gct_loss = compute_gct_loss(proj_gct, cls_token.detach())
  ```
- **Bài học rút ra:** Luôn kiểm tra kỹ mạch lan truyền ngược (Gradient Flow), bất kỳ vector tham chiếu nào từ Backbone đóng băng đều phải được `.detach()` để bảo vệ đặc trưng nền.

---

### 2️⃣ BUG 2: VRAM Memory Leak (Tràn bộ nhớ GPU qua 5000 iters)
- **Triệu chứng:** Huấn luyện trên GPU 8GB/16GB chạy tới khoảng iteration 1200 thì bị văng lỗi `CUDA Out of Memory (OOM)`.
- **Nguyên nhân root-cause:** Hàm tính `global_cosine_hm_percent` có sử dụng Gradient Hook để đăng ký tính gradient. Tuy nhiên, các hook này không được giải phóng sau mỗi bước `backward()`, khiến đồ thị tính toán (Computation Graph) bị tích tụ liên tục trong VRAM.
- **Cách khắc phục:** 
  Tự động gỡ bỏ hook ngay sau khi hoàn thành phép tính bằng lệnh `h.remove()`:
  ```python
  handle = tensor.register_hook(hook_fn)
  # ... compute loss ...
  handle.remove()  # Tự xóa hook giải phóng VRAM!
  ```
- **Bài học rút ra:** Khi viết các custom loss phức tạp có sử dụng PyTorch Hooks, phải luôn quản lý vòng đời và hủy hook để tránh rò rỉ bộ nhớ.

---

### 3️⃣ BUG 3: Spatial Coordinate Misalignment (Lệch tọa độ không gian sPRO)
- **Triệu chứng:** Chỉ số sPRO (Pixel-level localization) của GCT V2 ban đầu chỉ đạt `60.90%`, thấp hơn hẳn kỳ vọng mặc dù ảnh Heatmap nhìn rất đẹp.
- **Nguyên nhân root-cause:** Ảnh đầu vào $448 \times 448$ bị `CenterCrop` lấy $392 \times 392$ ở giữa. Khi mô hình tính Anomaly Map kích thước $28 \times 28$, em chỉ upsample thẳng lên $448 \times 448$ mà không bù lại lề (Padding offset 28px). Việc này làm bản đồ bất thường bị giãn và lệch tọa độ so với mặt nạ Ground Truth Mask $448 \times 448$.
- **Cách khắc phục:** 
  Upsample map về $392 \times 392$, sau đó dán vào giữa canvas zero $448 \times 448$ tại vị trí `canvas[28:420, 28:420]`.
- **Kết quả:** Khôi phục độ chuẩn xác sPRO lên **`70.10%`** (tăng **+9.20%**)!
- **Bài học rút ra:** Độ chính xác ở mức pixel đòi hỏi sự nhất quán 100% về hệ tọa độ giữa Preprocessing Transform và Evaluation Ground Truth.

---

### 4️⃣ BUG 4: PyTorch Hub Network Instability (`RemoteDisconnected`)
- **Triệu chứng:** Mỗi lần chạy `python src/eval.py` trên Cloud GPU (RunPod/Vast.ai), PyTorch Hub lại phát request kiểm tra trên GitHub và hay bị văng lỗi `RemoteDisconnected` hoặc `HTTP 429 Rate Limit`.
- **Nguyên nhân root-cause:** `torch.hub.load()` mặc định luôn kiểm tra phiên bản trên mạng mỗi khi mở phiên Python mới.
- **Cách khắc phục:** 
  Viết cơ chế phát hiện cache local: Nếu thư mục cache `facebookresearch_dinov2_main` đã tồn tại trên đĩa cứng, lập tức nạp bằng `source='local'`:
  ```python
  if hub_dir.exists():
      backbone = torch.hub.load(str(hub_dir), 'dinov2_vitb14_reg', source='local')
  ```
- **Kết quả:** Nạp tức thì trong **0.1 giây**, tắt hoàn toàn kết nối mạng (ZERO Network Calls)!
- **Bài học rút ra:** Hệ thống chạy thực nghiệm công nghiệp phải hỗ trợ chế độ Offline-First để đảm bảo tính ổn định và tốc độ.

---

### 5️⃣ BUG 5: Test-Set Tuning Bias ($\gamma$ Overfitting)
- **Triệu chứng:** Nếu quét tìm giá trị $\gamma$ tối ưu trên tập test (Oracle Sweep), ta có thể đẩy Mean AUROC lên đỉnh `87.27%` ($\gamma=0.8$). Nhưng việc này vi phạm nghiêm trọng phương pháp luận nghiên cứu (Data Leakage).
- **Cách khắc phục:** 
  Cố định hệ số $\gamma = 1.0$ (Fixed Coefficient Weighting) a-priori trước khi chạy tập test. Đưa con số **86.68% Mean AUROC** ($\gamma=1.0$) làm kết quả thực nghiệm chính thức, và giáng cấp con số 87.27% xuống làm "Oracle Upper Bound" trong phần phân tích độ nhạy.
- **Bài học rút ra:** Sự trung thực và chặt chẽ về mặt phương pháp luận nghiên cứu quan trọng hơn việc chạy theo con số đẹp một cách vô căn cứ.

---

### 6️⃣ BUG 6: Loss Explosion due to Hard-Mining Ratio $p$
- **Triệu chứng:** Ở 100 iterations đầu tiên khi mới huấn luyện, loss bị bộc phát giá trị rất lớn làm gradient bị nổ (Exploding Gradient).
- **Nguyên nhân root-cause:** Hệ số chọn patch khó $p=0.9$ được áp dụng ngay từ bước đầu tiên khi mô hình chưa học được gì, khiến nó bị ép tập trung vào các patch nhiễu mạnh nhất.
- **Cách khắc phục:** 
  Thêm cơ chế Linear Warmup cho hệ số $p$: cho $p$ tăng dần từ $0 \rightarrow 0.9$ trong 1000 iterations đầu tiên.
- **Bài học rút ra:** Các cơ chế Hard-negative Mining luôn cần bước khởi động mềm (Warmup) để mô hình định hình được không gian đặc trưng cơ bản trước khi nén khó.

---

### 7️⃣ BUG 7: Refactored Code Evaluation Crash (`NameError`)
- **Triệu chứng:** Khi refactor hàm `evaluate()` trong `src/eval.py`, script bị crash do thiếu khai báo các biến `category`, `img_size`, `target_layers`.
- **Cách khắc phục:** Đọc trực tiếp các tham số này từ dictionary cấu hình JSON `cfg["dataset"]["img_size"]` và bổ sung đầy đủ tham số đầu vào.
- **Bài học rút ra:** Luôn chạy thử nghiệm tích hợp (Integration Test) trên 1 category nhỏ trước khi chạy batch toàn bộ 5 categories trên GPU.
