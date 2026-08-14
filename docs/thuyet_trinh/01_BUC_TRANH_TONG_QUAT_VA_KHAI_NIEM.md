# 🌐 CHUYÊN ĐỀ 1: BỨC TRANH TỔNG QUAN VÀ CÁC KHÁI NIỆM CỐT LÕI

---

## 1. BỨC TRANH TỔNG QUAN (THE BIG PICTURE)

### ❓ Bài toán Phát hiện Bất thường Công nghiệp (Industrial Anomaly Detection) là gì?
- **Trực giác (Intuition):** Hãy tưởng tượng một dây chuyền sản xuất chai nước trái cây hay hộp đồ ăn sáng trong nhà máy chạy với tốc độ 100 sản phẩm/phút. Con người không thể đứng nhìn từng chai xem có bị vỡ hay thiếu nhãn không. Chúng ta cần một hệ thống camera thông minh (Computer Vision) tự động chụp ảnh và báo động ngay khi có sản phẩm bị lỗi.
- **Định nghĩa:** Phát hiện bất thường là bài toán phân loại (Classification) và định vị (Localization) các mẫu dữ liệu không tuân theo quy luật của các mẫu bình thường (Good/Normal samples) được học trong quá trình huấn luyện.
- **Thách thức đặc thù (Unsupervised Setting):** Trong nhà máy, chúng ta có hàng triệu sản phẩm **BÌNH THƯỜNG (Good images)** để huấn luyện, nhưng **KHÔNG CÓ HOẶC CÓ RẤT ÍT SẢN PHẨM LỖI (No anomaly images during training)** vì lỗi xảy ra rất hiếm và ngẫu nhiên. Do đó, mô hình chỉ được học trên ảnh BÌNH THƯỜNG.

---

## 2. PHÂN BIỆT RÕ: STRUCTURAL VS. LOGICAL ANOMALY (MVTec LOCO AD)

Tập dữ liệu **MVTec LOCO AD (Logical Constraints Anomaly Detection)** là benchmark chuẩn công nghiệp khó nhất hiện nay vì chia lỗi làm 2 nhóm riêng biệt:

```text
               ┌─────────────────────────────────────────────────────────┐
               │              MVTec LOCO AD Anomaly Types                │
               └────────────────────────────┬────────────────────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
     ┌─────────────────────────────┐                 ┌─────────────────────────────┐
     │ Structural Anomaly (Cấu trúc)│                 │   Logical Anomaly (Logic)   │
     ├─────────────────────────────┤                 ├─────────────────────────────┤
     │ • Vết xước, móp, thủng.     │                 │ • Thiếu linh kiện (thiếu ốc)│
     │ • Vết dơ, rách nhãn.        │                 │ • Sai thứ tự linh kiện.     │
     │ • Thay đổi bề mặt cục bộ.   │                 │ • Dừ lượng/thiếu dung tích. │
     └──────────────┬──────────────┘                 └──────────────┬──────────────┘
                    │                                               │
                    ▼                                               ▼
       [Phát hiện bằng Patch cục bộ]                    [CẦN NGỮ CẢNH ĐẠI CỤC GCT]
```

### A. Bất thường Cấu trúc (Structural Anomaly)
- **Trực giác:** Nhìn vào chai nước thấy có một vết nứt nhỏ trên vỏ chai.
- **Thuật ngữ:** `Structural Anomaly` / `Local Texture Defect`.
- **Định nghĩa:** Là sự thay đổi bất thường về mặt thị giác tại một vùng không gian nhỏ (Local Region/Patch), biến đổi kết cấu bề mặt (Texture/Color) so với mẫu chuẩn.
- **Ví dụ dry-run:** 
  - Ảnh chuẩn: Mọi pixel tại vị trí vết xước đều có màu vàng nhạt (giá trị $[200, 200, 200]$).
  - Ảnh lỗi: Xuất hiện vệt màu đen xước (giá trị $[10, 10, 10]$).
  - Khoảng cách Cosine hoặc MSE giữa ảnh tái tạo và ảnh gốc tại đúng pixel đó tăng vọt!

### B. Bất thường Logic (Logical Anomaly)
- **Trực giác:** Hộp đồ ăn sáng có 2 ngăn kẹo và 1 ngăn bánh. Ảnh lỗi có 3 ngăn kẹo và 0 ngăn bánh, mặc dù từng viên kẹo và từng chiếc bánh đều **HOÀN HẢO KHÔNG MỘT VẾT XƯỚC**!
- **Thuật ngữ:** `Logical Anomaly` / `Global Constraint Defect`.
- **Định nghĩa:** Là sự vi phạm các quy tắc/ràng buộc logic mang tính hệ thống toàn ảnh (Global Logic/Constraints), ví dụ như sai số lượng, sai vị trí tương quan, hoặc thừa/thiếu linh kiện, mặc dù bề mặt chi tiết của từng linh kiện hoàn toàn bình thường.
- **Tại sao mô hình cũ (Baseline Dinomaly) thất bại ở đây?**
  - Mô hình baseline tái tạo ảnh dựa trên các ô vuông nhỏ cục bộ (`Patch Tokens` $14 \times 14$). Khi nhìn vào từng ô vuông chứa viên kẹo, mô hình thấy viên kẹo rất đẹp nên nó tái tạo lại viên kẹo hoàn hảo. Do đó, điểm lỗi cục bộ ($\text{Score}_{\text{patch}}$) cực kỳ thấp $\rightarrow$ **BỎ SÓT LỖI LOGIC!**

---

## 3. GIẢI PHÁP ĐỀ XUẤT: GLOBAL CONSISTENCY TOKEN (GCT)

### A. Trực giác (Intuition)
Để không bị lừa bởi các viên kẹo đẹp, chúng ta cần một **"Giám sát viên Quản lý Tổng thể"** (GCT Token). Người giám sát này không nhìn chi tiết từng viên kẹo, mà nhìn toàn bộ bức tranh để đếm: "Hộp này có đúng 2 kẹo + 1 bánh không?". Nếu sai quy luật toàn cục, người giám sát sẽ phát còi báo động ngay lập tức!

### B. Thuật ngữ gốc (English Terminology)
- **Learned Context Token:** Token ngữ cảnh học được.
- **Active Dual-Stream Scoring:** Cơ chế chấm điểm kích hoạt luồng đôi.
- **Cosine Distance Alignment Loss:** Hàm mất mát khoảng cách Cosine căn chỉnh ngữ cảnh.

### C. Định nghĩa Toán học & Thuật toán
GCT Token ($\mathbf{t}_{\text{gct}} \in \mathbb{R}^{768}$) là 1 vector tham số học được, được chèn vào chuỗi Patch Tokens sau lớp Bottleneck:

$$\mathbf{Z} = [\mathbf{t}_{\text{gct}}, \mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N] \in \mathbb{R}^{(1 + N) \times 768}$$

GCT Token tương tác với toàn bộ $N=784$ patch tokens thông qua 8 lớp Linear Attention Decoder. Sau đó, vector đầu ra $\mathbf{t}_{\text{gct}}^{\text{out}}$ được đưa qua một Projection Head:

$$\mathbf{p}_{\text{gct}} = \text{LayerNorm}(\text{Linear}(\mathbf{t}_{\text{gct}}^{\text{out}})) \in \mathbb{R}^{768}$$

Hàm Loss GCT thúc đẩy $\mathbf{p}_{\text{gct}}$ tiến sát đến vector CLS token đại cục của DINOv2 ($\mathbf{c}_{\text{dino}} \in \mathbb{R}^{768}$):

$$\mathcal{L}_{\text{GCT}} = 1.0 - \frac{\mathbf{p}_{\text{gct}} \cdot \mathbf{c}_{\text{dino}}}{\|\mathbf{p}_{\text{gct}}\|_2 \|\mathbf{c}_{\text{dino}}\|_2}$$

Khi suy luận (Inference), điểm bất thường cuối cùng là sự kết hợp luồng đôi (Dual-Stream):

$$\text{Score}_{\text{final}} = \text{Score}_{\text{patch}} + \gamma \cdot \text{Score}_{\text{GCT}}$$

Trong đó $\text{Score}_{\text{GCT}} = \mathcal{L}_{\text{GCT}}$ chính là giá trị khoảng cách Cosine đại cục.

---

## 4. TÍNH TOÁN BẰNG TAY (DRY RUN EXAMPLE)

Hãy làm một ví dụ tính toán bằng tay trên ma trận $2 \times 1$ để thấy rõ cách GCT phát hiện lỗi logic:

### Giả sử:
- Vector đại cục của DINOv2 đại diện cho trạng thái chuẩn "Hộp 2 kẹo + 1 bánh":
  $$\mathbf{c}_{\text{dino}} = [1.0, 0.0]^T$$
- **Trường hợp 1 (Ảnh bình thường - Good image):**
  - Mô hình GCT trích xuất ra vector: $\mathbf{p}_{\text{gct}} = [0.99, 0.1]^T$
  - Khoảng cách Cosine ($\text{Score}_{\text{GCT}}$):
    $$\text{CosSim} = \frac{1.0 \times 0.99 + 0.0 \times 0.1}{\sqrt{1^2 + 0^2} \times \sqrt{0.99^2 + 0.1^2}} = \frac{0.99}{1.0 \times 0.995} \approx 0.995$$
    $$\text{Score}_{\text{GCT}} = 1.0 - 0.995 = 0.005 \quad (\text{RẤT THẤP} \rightarrow \text{BÌNH THƯỜNG})$$

- **Trường hợp 2 (Ảnh lỗi logic - Logical Anomaly: Hộp 3 kẹo + 0 bánh):**
  - Mô hình GCT trích xuất ra vector: $\mathbf{p}_{\text{gct}} = [0.5, 0.866]^T$ (do bộc phát sự lệch ngữ cảnh)
  - Khoảng cách Cosine ($\text{Score}_{\text{GCT}}$):
    $$\text{CosSim} = \frac{1.0 \times 0.5 + 0.0 \times 0.866}{1.0 \times \sqrt{0.5^2 + 0.866^2}} = \frac{0.5}{1.0} = 0.5$$
    $$\text{Score}_{\text{GCT}} = 1.0 - 0.5 = 0.500 \quad (\text{BỘC PHÁT TĂNG 100 LẦN} \rightarrow \text{BÁO ĐỘNG BẤT THƯỜNG!})$$

---

## 5. MAPPING TRỰC TIẾP VÀO FILE CODE PROJECT

- Khởi tạo GCT Token & Projection Head: [`src/models/vitill_gct.py`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/models/vitill_gct.py#L170-L195)
- Hàm tính GCT Loss: [`src/losses/gct_loss.py`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/losses/gct_loss.py#L15-L35)
- Công thức Dual-Stream Active Scoring: [`src/eval.py`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/eval.py#L110-L135)
