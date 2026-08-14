# 🎤 CHUYÊN ĐỀ 5: KỊCH BẢN BÁO CÁO GIỮA KỲ NGÀY MAI VÀ BỘ 10 CÂU HỎI PHẢN BIỆN (DEFENSE SCRIPT & Q&A)

---

## 🕒 KỊCH BẢN NÓI CHI TIẾT THEO TỪNG PHÚT (5-MINUTE PRESENTATION)

### 📌 Phút 0:00 - 1:00: Mở đầu & Nối tiếp tiến độ đợt trước
> *"Em chào Thầy/Cô ạ! Hôm nay em xin báo cáo tiến độ đồ án Giữa kỳ nối tiếp từ buổi trao đổi lần trước ạ.*
>
> *Ở buổi trước, em đã thiết lập xong Dinomaly Baseline đạt Mean AUROC 84.69%. Nhưng khi thử nghiệm GCT V1 (Passive token), kết quả bị đứng chân ở 84.64% (không tăng). Hôm đó em và Thầy có thảo luận giả thuyết: do MLP Projection Head 2 lớp đang bị hấp thụ gradient, và tín hiệu 1 token bị patch tokens lấn át.*
>
> *Từ buổi đó đến nay, em đã tập trung triển khai hướng khắc phục này và thu được kết quả rất khả quan ạ."*

---

### 📌 Phút 1:00 - 2:30: Trình bày cải tiến GCT V2 & Bảng số liệu chính
*(Hành động: Mở màn hình file [`docs/presentation_summary.md`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/docs/presentation_summary.md))*

> *"Dạ, đúng như hướng chỉ đạo của Thầy, em đã thực hiện 2 cải tiến cốt lõi:*
> 1. *Thay MLP 2 lớp bằng 1 lớp **Linear + LayerNorm** để giảm hấp thụ gradient.*
> 2. *Tích hợp cơ chế **Active Dual-Stream Scoring** ($\text{Score}_{\text{final}} = \text{Score}_{\text{patch}} + 1.0 \cdot \text{Score}_{\text{GCT}}$) trực tiếp lúc suy luận.*
>
> *Kết quả thực nghiệm trên 5 category sản phẩm MVTec LOCO AD như sau ạ:*
> - **Logical Anomaly AUROC (Lỗi Logic):** Tăng mạnh từ **76.40% lên 80.33%** (Tăng **+3.93%** so với Baseline). Đặc biệt trên tập `screw_bag` (ốc vít rời) tăng kỷ lục **+9.44%** (từ 59.19% lên 68.63%).
> - **Mean AUROC:** Nâng từ **84.67% lên 86.68%** (Tăng **+2.01%**).
> - **Structural AUROC (Lỗi cấu trúc):** Duy trì tương đương Baseline (**93.04%** vs 92.97%).
> - **Tốc độ Inference:** Đạt **87.6 ms/ảnh (~11.4 FPS)** trên GPU, đáp ứng thời gian thực."*

---

### 📌 Phút 2:30 - 4:00: Trình bày phân tích sâu & Căn chỉnh tọa độ không gian
*(Hành động: Cho Thầy xem ảnh Heatmap minh họa `docs/figures/heatmap_screw_bag_logical_anomalies_1.png`)*

> *"Em xin giải thích ngắn gọn lý do tại sao GCT V2 lại hiệu quả ạ:*
> - *Với lỗi cấu trúc (vết xước), luồng Patch cục bộ tái tạo đã làm rất tốt.*
> - *Nhưng với lỗi logic (thiếu ốc, sai thứ tự), từng patch ô vuông nhỏ nhìn đều hoàn hảo nên luồng Patch bị bỏ sót. Lúc này, **GCT Token đóng vai trò tổng hợp ngữ cảnh đại cục toàn ảnh**. Em đo thử phân phối thì thấy `Score_GCT` ở ảnh bình thường rất thấp ($\approx 0.002$), nhưng khi gặp lỗi logic thì bộc phát tăng vọt (đạt đỉnh $0.1434$), đóng vai trò như còi báo động toàn cục.*
> - *Đồng thời, em cũng đã fix xong lỗi lệch tọa độ sPRO bằng cách dán căn chỉnh CenterCrop ($392 \rightarrow 448$), đưa sPRO định vị pixel đạt **70.10%** (tương đương Baseline 70.53%)."*

---

### 📌 Phút 4:00 - 5:00: Kết luận & Xin ý kiến cho Cuối kỳ
> *"Tóm lại, tiến độ giữa kỳ của em đã xác minh thành công hiệu quả của GCT V2 trên bài toán Logical Anomaly. Em đã lưu trữ mã nguồn và lịch sử thực nghiệm sạch sẽ trên Git.*
>
> *Em rất mong nhận được thêm ý kiến đóng góp của Thầy để em tiếp tục hoàn thiện báo cáo Cuối kỳ ạ. Em cảm ơn Thầy nhiều ạ!"*

---

## ❓ BỘ 10 CÂU HỎI PHẢN BIỆN CỦA THẦY VÀ ĐÁP ÁN MẪU XUẤT SẮC

### ❓ Q1: "Tại sao Logical AUROC tăng mạnh (+3.93%) mà Structural AUROC gần như không đổi?"
- **Đáp án:** *"Dạ Thầy, lỗi Structural (vết xước) chỉ xuất hiện ở các ô vuông không gian nhỏ, luồng Patch tái tạo cục bộ đã giải quyết rất tốt rồi ạ. Còn lỗi Logical (thiếu linh kiện) đòi hỏi hiểu biết tương quan không gian đại cục toàn ảnh — đây chính là điểm yếu của luồng Patch mà GCT Token được thiết kế để bù đắp. Vì vậy GCT V2 tập trung bứt phá mạnh ở Logical AUROC mà không làm ảnh hưởng đến Structural AUROC ạ."*

---

### ❓ Q2: "Tại sao em lại chọn trọng số $\gamma = 1.0$ mà không chọn số khác?"
- **Đáp án:** *"Dạ Thầy, giá trị $\gamma = 1.0$ được em cố định a-priori (trước khi đánh giá tập test) để đảm bảo tính khách quan khoa học, tránh hiện tượng test-set tuning bias (Data Leakage). Nếu em dùng hàm quét Oracle sweep chọn $\gamma$ tối ưu từng category trên tập test thì kết quả có thể đạt đỉnh 87.27%, nhưng em quyết định lấy con số chuẩn 86.68% ($\gamma=1.0$) để làm báo cáo trung thực ạ."*

---

### ❓ Q3: "GCT Token tương tác với Patch Tokens như thế nào trong Decoder?"
- **Đáp án:** *"Dạ Thầy, GCT Token được ghép nối vào đầu chuỗi Patch Tokens thành ma trận $[B, 785, 768]$ trước khi đi qua 8 lớp Linear Attention Decoder. Trong mỗi lớp Decoder, GCT Token đóng vai trò là một query/key/value token cùng tham gia vào phép nhân chú ý (Attention), giúp nó thu thập thông tin đại cục từ tất cả 784 patch tokens còn lại ạ."*

---

### ❓ Q4: "Tại sao em lại dùng DINOv2-Register thay vì DINOv2 thường?"
- **Đáp án:** *"Dạ Thầy, bản DINOv2 thông thường có nhược điểm bị nhiễu các điểm cực trị vô nghĩa (Artifact Spikes) ở vùng nền tĩnh. Bản `-Register` có thêm 4 Register Tokens đóng vai trò làm hố hút rác nhiễu, giúp các Patch Tokens giữ được đặc trưng cực kỳ sạch và ổn định ạ."*

---

### ❓ Q5: "Tại sao em lại phải dùng hàm `.detach()` ở DINOv2 CLS Token?"
- **Đáp án:** *"Dạ Thầy, DINOv2 Backbone được em đóng băng hoàn toàn (`requires_grad=False`) để giữ nguyên không gian đặc trưng tổng quát. Khi tính GCT Loss giữa Projection Head và CLS token, nếu không `.detach()` thì gradient từ GCT loss sẽ trôi ngược về làm hỏng trọng số backbone ạ."*

---

### ❓ Q6: "Tại sao em lại chọn Linear Attention thay vì Softmax Attention chuẩn?"
- **Đáp án:** *"Dạ Thầy, Softmax Attention có độ phức tạp tính toán theo bình phương $O(N^2)$ với $N=784$ patch tokens. Dùng Linear Attention (ELU+1) giảm độ phức tạp xuống $O(N)$, giúp tốc độ inference của mô hình đạt **87.6 ms/ảnh (~11.4 FPS)** trên GPU, đáp ứng chuẩn thời gian thực ạ."*

---

### ❓ Q7: "Tại sao sPRO của GCT V2 (70.10%) lại tương đương Baseline (70.53%) mà không cao hơn hẳn?"
- **Đáp án:** *"Dạ Thầy, sPRO là chỉ số đo độ đè phủ ở cấp độ pixel (Pixel-level localization). Bản thân GCT Token là tín hiệu phân loại đại cục (Global Image-level Trigger) chứ không làm thay đổi bản đồ tái tạo patch cục bộ. Do đó GCT V2 tập trung nâng cao khả năng phát hiện ảnh lỗi (Image-level AUROC) trong khi vẫn duy trì chuẩn xác khả năng khoanh vùng bất thường (Pixel-level sPRO) của Baseline ạ."*

---

### ❓ Q8: "Em đã xử lý lỗi tràn VRAM GPU khi huấn luyện ra sao?"
- **Đáp án:** *"Dạ Thầy, do hàm loss hard-mining có sử dụng PyTorch Gradient Hooks để đăng ký tính toán. Ban đầu các hook này không được tự hủy làm VRAM tăng dần qua 5000 iters. Em đã bổ sung lệnh `handle.remove()` tự động giải phóng hook ngay sau bước backward, giải quyết triệt để lỗi tràn VRAM ạ."*

---

### ❓ Q9: "Em đã giải quyết lỗi ngắt mạng khi nạp DINOv2 trên Cloud GPU thế nào?"
- **Đáp án:** *"Dạ Thầy, PyTorch Hub mặc định luôn gọi request ra internet để kiểm tra. Em đã viết cơ chế kiểm tra cache đĩa cứng local, nếu đã có thư mục `facebookresearch_dinov2_main` thì tự động nạp bằng tham số `source='local'`, giúp mô hình nạp tức thì trong 0.1s và tắt 100% kết nối mạng ạ."*

---

### ❓ Q10: "Kế hoạch tiếp theo của em cho Báo cáo Cuối kỳ là gì?"
- **Đáp án:** *"Dạ Thầy, cho giai đoạn cuối kỳ, em dự định sẽ bổ sung đánh giá đa seed (Multi-seed validation trên 3 seeds 42, 123, 2025) để tính độ lệch chuẩn (Std), hoàn thiện các biểu đồ trực quan hóa bổ sung và viết hoàn chỉnh các chương của cuốn Báo cáo Đồ án ạ!"*
