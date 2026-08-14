# 🗺️ LỘ TRÌNH MASTER DỰ ÁN: TỪ ZERO ĐẾN BẢO VỆ & ĐI PHỎNG VẤN CV (MASTER INDEX)

---

## 💡 GIỚI THIỆU LỘ TRÌNH HỌC (LEARNING ROADMAP)

Tài liệu này được thiết kế theo dạng **Masterclass Cầm Tay Chỉ Việc**, dành cho người chưa nắm sâu về mã nguồn nhưng muốn hiểu tận gốc dự án **"Industrial Anomaly Detection on MVTec LOCO AD using Dinomaly + Global Consistency Token (GCT)"**.

Mục tiêu: Đảm bảo bạn **nắm vững bức tranh tổng thể, tự tin giải thích từng dòng code, thuộc lòng thuật toán chạy bằng tay, hiểu rõ nguyên nhân từng lỗi bug đã fix**, và **tự tin 100% khi báo cáo trước Hội đồng cũng như trả lời phỏng vấn tuyển dụng AI/Computer Vision Engineer**.

---

## 📂 DANH SÁCH 6 CHUYÊN ĐỀ TÀI LIỆU CHUYÊN SÂU

| File Tài Liệu | Nội Dung Chính | Mục Tiêu Đạt ĐƯỢC |
|:---|:---|:---|
| [`01_BUC_TRANH_TONG_QUAT_VA_KHAI_NIEM.md`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/docs/thuyet_trinh/01_BUC_TRANH_TONG_QUAT_VA_KHAI_NIEM.md) | Bức tranh tổng quan bài toán, phân biệt Structural vs. Logical Anomaly, Intuition $\rightarrow$ Định nghĩa $\rightarrow$ Chạy tay thuật toán. | Nắm được "tại sao dự án này tồn tại" và bài toán MVTec LOCO AD giải quyết vấn đề gì. |
| [`02_KIEN_TRUC_MO_HINH_CHI_TIET.md`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/docs/thuyet_trinh/02_KIEN_TRUC_MO_HINH_CHI_TIET.md) | Phân tích chi tiết 5 khối: DINOv2-Register, Bottleneck MLP, Decoder Linear Attention, GCT Token & Projection Head. | Hiểu rõ luồng dữ liệu (Dataflow) và ma trận kích thước Tensor ($B, N, D$) qua từng lớp. |
| [`03_MAPPING_CODE_LINE_BY_LINE.md`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/docs/thuyet_trinh/03_MAPPING_CODE_LINE_BY_LINE.md) | Mapping 1:1 từng dòng code trong `vitill_gct.py`, `cosine_loss.py`, `eval.py`, `train.py`. Input, Output, Syntax, Rationale. | Đọc code hiểu ngay chức năng từng dòng, không sợ Thầy hay Nhà tuyển dụng chỉ vào code hỏi. |
| [`04_NHAT_KY_SUA_LOI_VA_BAI_HOC.md`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/docs/thuyet_trinh/04_NHAT_KY_SUA_LOI_VA_BAI_HOC.md) | Tổng hợp 7 lỗi bug thực tế kinh điển (Gradient Leak, Memory Leak, Spatial Alignment $392 \rightarrow 448$, PyTorch Hub Cache, Test-set bias). | Trả lời tự tin câu hỏi "Em đã gặp khó khăn/bug gì khi làm dự án và giải quyết ra sao?". |
| [`05_KICH_BAN_BAO_CAO_GIUA_KY_NGAY_MAI.md`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/docs/thuyet_trinh/05_KICH_BAN_BAO_CAO_GIUA_KY_NGAY_MAI.md) | Kịch bản nói từng phút cho báo cáo giữa kỳ ngày mai, cách show màn hình, và Bộ 10 câu hỏi xoáy của Thầy kèm đáp án. | Báo cáo trôi chảy, đúng phong thái sinh viên làm tiến độ bài bản, đạt điểm số cao nhất. |
| [`06_BO_CAU_HOI_PHONG_VAN_CV.md`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/docs/thuyet_trinh/06_BO_CAU_HOI_PHONG_VAN_CV.md) | Bộ 15 câu hỏi phỏng vấn tuyển dụng AI/CV Engineer (Technical Deep-dive, System Trade-offs, Future Improvements). | Viết dự án này vào CV và tự tin chinh phục nhà tuyển dụng các vị trí AI/CV. |
| [`07_TAI_LIEU_THAM_KHAO_VA_NGUON_CODE.md`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/docs/thuyet_trinh/07_TAI_LIEU_THAM_KHAO_VA_NGUON_CODE.md) | Danh sách 4 bài báo khoa học cốt lõi (Dinomaly CVPR 2025, DINOv2 Meta AI, MVTec LOCO AD, Linear Attention) và nguồn mã nguồn. | Trả lời tự tin câu hỏi "Em đã đọc và kế thừa ý tưởng từ bài báo / nguồn code nào?". |

---

## 🎯 QUY TẮC NÂNG CAO NĂNG LỰC HỌC (LEARNING METHODOLOGY)

Mọi khái niệm trong bộ tài liệu này đều tuân thủ nghiêm ngặt theo **Cấu trúc 5 Bước Học Sâu**:
1. **Trực Giác (Intuition):** Tương tự đời sống thực tế giúp não bộ tiếp thu tự nhiên.
2. **Thuật Ngữ Gốc (English Terminology):** Tên gọi chuẩn trong các bài báo khoa học TOP-1 (CVPR, ECCV, NeurIPS).
3. **Định Nghĩa Chính Xác (Formal Definition):** Công thức toán học và nguyên lý hoạt động.
4. **Chạy Tay Thuật Toán (Dry Run Example):** Mô phỏng tính toán từng số lẻ trên ma trận nhỏ $2 \times 2$.
5. **Mapping Code Trực Tiếp (Code Line Mapping):** Chỉ rõ đoạn code nào thực thi khái niệm đó trong dự án.
