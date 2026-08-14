# 📊 HƯỚNG DẪN CHẠY & GIẢI THÍCH CÁC TIÊU CHÍ ĐÁNH GIÁ (EVALUATION GUIDE)
## Dự án: ViTill-GCT V2 trên MVTec LOCO AD Benchmark

---

## 🗺️ TỔNG QUAN 5 TIÊU CHÍ ĐÁNH GIÁ METRICS

Dự án đánh giá mô hình trên **5 tiêu chí chuẩn công nghiệp** dành riêng cho tập dữ liệu MVTec LOCO AD:

| STT | Tiêu Chí (Metric) | Cấp Độ (Level) | Công Thức / Nguyên Lý | Ý Nghĩa Thực Tế |
|:---:|:---|:---:|:---|:---|
| **1** | **Logical AUROC (%)** | Image-level | ROC-AUC trên tập `logical_anomalies` vs `good` | Đo khả năng phát hiện lỗi ngữ cảnh đại cục (thiếu/thừa/sai vị trí linh kiện). |
| **2** | **Structural AUROC (%)** | Image-level | ROC-AUC trên tập `structural_anomalies` vs `good` | Đo khả năng phát hiện lỗi bề mặt cục bộ (vết xước, biến dạng, nứt nẻ). |
| **3** | **Mean AUROC (%)** | Image-level | $\text{Mean} = \frac{\text{Logical AUROC} + \text{Structural AUROC}}{2}$ | Chỉ số tổng hợp hiệu năng bắt bất thường toàn diện của mô hình. |
| **4** | **sPRO (%)** | Pixel-level | Structural Pseudo-ROC với ngưỡng $\text{FPR} \in [0, 0.30]$ | Đo độ chính xác khoanh vùng lỗi pixel theo từng connected component. |
| **5** | **Latency (ms/ảnh)** | System-level | Thời gian xử lý trung bình 1 ảnh khi $\text{batch\_size} = 1$ | Kiểm tra khả năng đáp ứng thời gian thực (Real-time) trên dây chuyền nhà máy. |

---

## 💻 1. CÁC LỆNH CHẠY ĐÁNH GIÁ (EVAL COMMANDS)

Tất cả các đánh giá đều được thực hiện qua file [`src/eval.py`](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/eval.py).

### 🔹 1.1. Đánh giá Mô hình Baseline (Không dùng GCT)
Chạy đánh giá mô hình Baseline trên 1 category (ví dụ: `screw_bag`):
```bash
python src/eval.py --category screw_bag
```

---

### 🔹 1.2. Đánh giá Mô hình ViTill-GCT V2 (Có GCT Module)
Chạy đánh giá mô hình ViTill-GCT V2 với cờ `--use_gct`:
```bash
python src/eval.py --category screw_bag --use_gct
```

---

### 🔹 1.3. Tùy chỉnh Trọng số Dual-Stream $\gamma$ (Active Scoring Weight)
Công thức điểm số lúc inference:
$$\text{Score}_{\text{final}} = \text{Score}_{\text{patch}} + \gamma \cdot \text{Score}_{\text{GCT}}$$

Mặc định $\gamma = 1.0$. Nếu muốn thử nghiệm các giá trị $\gamma$ khác (ví dụ: $\gamma = 0.5$ hoặc $\gamma = 1.5$):
```bash
python src/eval.py --category screw_bag --use_gct --gamma 0.5
```

---

### 🔹 1.4. Đánh giá trên CPU (Nếu không có GPU CUDA)
Thêm cờ `--cpu` để chạy đánh giá trên bộ vi xử lý CPU:
```bash
python src/eval.py --category screw_bag --use_gct --cpu
```

---

## 🔄 2. SCRIPT CHẠY TỰ ĐỘNG TOÀN BỘ 5 CATEGORIES

Để đánh giá tự động trên cả 5 sản phẩm của MVTec LOCO AD (`breakfast_box`, `juice_bottle`, `pushpins`, `screw_bag`, `splicing_connectors`):

### 🪟 Trên Windows (PowerShell)
```powershell
$categories = @("breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors")

Write-Host "=================== EVALUATING GCT V2 ===================" -ForegroundColor Green
foreach ($cat in $categories) {
    Write-Host "-> Category: $cat" -ForegroundColor Cyan
    python src/eval.py --category $cat --use_gct
}
```

---

### 🐧 Trên Linux / Cloud GPU (Bash Script)
```bash
for cat in breakfast_box juice_bottle pushpins screw_bag splicing_connectors; do
    echo "=================== EVALUATING GCT V2: $cat ==================="
    python src/eval.py --category $cat --use_gct
done
```

---

## 📈 3. CÁCH ĐỌC VÀ PHÂN TÍCH LOG OUTPUT KẾT QUẢ

Khi chạy thành công lệnh `python src/eval.py --category screw_bag --use_gct`, bạn sẽ nhận được log output như sau:

```text
[EVAL] Model: DINOMALY + GCT V2  |  Category: screw_bag  |  Device: cuda  |  Gamma: 1.0
[CKPT] Loaded checkpoint from: experiments/checkpoints/gct/gct_screw_bag_strict.pth
[BACKBONE] Loading DINOv2-Register from local hub cache (facebookresearch_dinov2_main)...
[BACKBONE] DINOv2-Register loaded and frozen.
[METRICS] Logical AUROC   : 68.63%
[METRICS] Structural AUROC: 93.04%
[METRICS] Mean AUROC      : 80.84%
[LATENCY] Batch=1 Inference Latency: 87.12 ms/image
[METRICS] sPRO (pixel-level): 70.10%
```

### 🔍 Giải thích các con số log:
1. **`Logical AUROC: 68.63%`**: Khả năng nhận diện ảnh bị thiếu/thừa ốc vít trong túi `screw_bag`. (Tăng **+9.44%** so với Baseline 59.19%).
2. **`Structural AUROC: 93.04%`**: Khả năng nhận diện vết trầy xước trên thân túi. (Bằng với Baseline 92.97%).
3. **`Mean AUROC: 80.84%`**: Trung bình cộng 2 chỉ số AUROC trên.
4. **`Batch=1 Inference Latency: 87.12 ms/image`**: Tốc độ xử lý 1 ảnh thực tế $\approx 11.4 \text{ FPS}$ trên GPU RTX.
5. **`sPRO (pixel-level): 70.10%`**: Độ bao phủ chính xác vị trí lỗi mức pixel ở giới hạn $\text{FPR} \le 30\%$.

---

## 🧩 4. MAPPING TRỰC TIẾP TỚI CODE TRONG `src/eval.py`

| Tiêu Chí | Hàm Xử Lý Trong Code | Dòng Code | Cơ Chế Kỹ Thuật Cốt Lõi |
|:---|:---|:---:|:---|
| **AUROC** | `compute_auroc(labels, scores)` | [`eval.py` L24-38](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/eval.py#L24-L38) | Gọi `sklearn.metrics.roc_auc_score` để tính diện tích dưới đường cong ROC. |
| **Anomaly Map** | `compute_anomaly_map(en_list, de_list)` | [`eval.py` L44-74](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/eval.py#L44-L74) | Tính Cosine Distance $1 - \text{CosSim}$, upsample về $392 \times 392$, dán canvas $448 \times 448$ offset 28px, lọc mượt Gaussian $\sigma=4$. |
| **Top-1% Score** | `image_score(anomaly_map)` | [`eval.py` L80-84](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/eval.py#L80-L84) | Lấy trung bình 2007 pixel có khoảng cách lớn nhất (Top 1%) để tránh lừa điểm cực trị. |
| **Dual-Stream** | `infer_one(...)` | [`eval.py` L90-116](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/eval.py#L90-L116) | Tổng hợp $\text{Score}_{\text{final}} = \text{Score}_{\text{patch}} + \gamma \cdot \text{Score}_{\text{GCT}}$. |
| **sPRO Metric** | `compute_spro(...)` | [`eval.py` L122-167](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/eval.py#L122-L167) | Quét 100 ngưỡng $\text{FPR} \in [0, 0.30]$ để tính diện tích bao phủ vùng lỗi theo connected component. |
| **Latency** | `evaluate(...)` | [`eval.py` L236-253](file:///C:/Users/mminh/OneDrive/Desktop/DAN/industrial-anomoly-detection/src/eval.py#L236-L253) | Warmup 5 lần, đồng bộ GPU `torch.cuda.synchronize()`, đo thời gian trung bình 20 lần chạy. |

---

## 🎯 BẢNG SO SÁNH KẾT QUẢ 5 CATEGORIES (BASELINE vs GCT V2)

| Category | Logical (Baseline) | Logical (GCT V2) | Gain Logical | Mean AUROC (Baseline) | Mean AUROC (GCT V2) | sPRO (GCT V2) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Breakfast Box** | 70.25% | **73.41%** | **+3.16%** | 79.80% | **81.38%** | 68.45% |
| **Juice Bottle** | 90.33% | **91.76%** | **+1.43%** | 93.90% | **94.62%** | 76.20% |
| **Pushpins** | 86.15% | **88.23%** | **+2.08%** | 88.07% | **89.11%** | 71.30% |
| **Screw Bag** | 59.19% | **68.63%** | 🔥 **+9.44%** | 76.08% | **80.84%** | 70.10% |
| **Splicing Connectors** | 76.08% | **79.62%** | **+3.54%** | 85.50% | **87.47%** | 64.45% |
| **MEAN 5 CATS** | **76.40%** | **80.33%** | 🏆 **+3.93%** | **84.67%** | **86.68%** | **70.10%** |

---

## 💡 LỜI KHUYÊN KHI TRẢ LỜI CÂU HỎI ĐÁNH GIÁ CỦA THẦY

1. **Thầy hỏi: "Tại sao Logical AUROC tăng đến +9.44% ở tập Screw Bag?"**
   - **Trả lời:** *"Dạ Thầy, tập `screw_bag` là tập túi ốc vít rời. Lỗi ở đây là đếm sai số lượng ốc trong túi — đây là lỗi ngữ cảnh đại cục thuần túy. GCT Token được thiết kế để học đại diện ngữ cảnh từ DINOv2 CLS token, kết hợp với luồng điểm `Score_GCT` bùng nổ khi số ốc bị sai, nên cải thiện cực kỳ rõ rệt ạ."*

2. **Thầy hỏi: "sPRO khác gì với Pixel AUROC thông thường?"**
   - **Trả lời:** *"Dạ Thầy, Pixel AUROC thông thường tính trên tất cả pixel của ảnh, dẫn đến việc các vết lỗi cực lớn áp đảo làm sai lệch điểm số. sPRO tính tỉ lệ đè đúng mask theo từng vùng lỗi liên thông (connected component) trong khoảng FPR $\le 30\%$, đo chính xác khả năng khoanh vùng lỗi thực tế ạ."*
