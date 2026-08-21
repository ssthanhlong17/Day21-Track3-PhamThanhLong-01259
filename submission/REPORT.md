# Lab 21 — Evaluation Report

**Họ tên**: Phạm Thanh Long  **MSSV**: 01259  **Ngày**: 2026-08-21
**Tier**: `CPU`  **Base model**: `Qwen/Qwen3.5-0.8B`  **GPU thực tế**: `CPU (không GPU)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage (mặc định) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 512 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 |

**Template có giữ khối ` thinking` không?** `có` — *(results/template_check.json)*
Template Qwen3.5 giữ nguyên khối suy luận trong `apply_chat_template`. Điều này có nghĩa là nếu dữ liệu huấn luyện chứa reasoning traces, chúng sẽ tới được hàm loss.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `0.39` |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
 response

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

> ⚠ **Cần GPU để chạy NB2.** Máy hiện tại là CPU-only, nên phần này chưa chạy được.
> Cần chạy trên Colab Free T4 theo hướng dẫn trong README.md.

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | _chưa đo_ | _chưa đo_ | _chưa đo_ | _chưa đo_ |
| (b) base + optimized prompt | _chưa đo_ | _chưa đo_ | _chưa đo_ | _chưa đo_ |
| (c) LoRA fine-tune | _chưa đo_ | _chưa đo_ | _chưa đo_ | _chưa đo_ |

**(b) có thật sự mạnh hơn (a) không?** _chưa đo_ — cần chạy NB2 trên GPU.
Bạn có sửa `OPTIMIZED_PROMPT` không? Không — giữ nguyên prompt mặc định của lab.

---

## 4. Giải phẫu cấu hình sai (NB4)

> ⚠ **Cần GPU để chạy NB4.** Phần này chưa chạy được trên máy CPU-only.

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | | | | | | |
| `attn_only` | q,v | *(matched)* | | | | | | |
| `wrong_lr` | text-linear | 16 | | | | | | |
| `qlora` | text-linear | 16 | | | | | | |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**

_Chưa trả lời được — cần chạy NB4 và NB5 trên GPU. Dựa trên lý thuyết deck §10.2,
attention-only placement với rank được nâng lên để khớp ngân sách tham số thường thua
full placement, vì rank không phải đòn bẩy chính — vị trí gắn adapter mới là yếu tố
quyết định. Nhưng cần số đo thực tế để xác nhận._

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

_Chưa trả lời được — cần chạy NB4 trên GPU. Theo deck §10.3, LR thang full-FT (1e-5)
áp cho LoRA sẽ làm loss gần như phẳng từ step 0, vì LoRA cần LR cao hơn ~10x. Nếu chỉ
nhìn loss mà không biết LR, có thể kết luận nhầm rằng "LoRA không học được" thay vì
"LR sai thang"._

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**

_Chưa trả lời được — cần chạy NB4 trên GPU. Nhà cung cấp (Unsloth) khuyến nghị không
dùng QLoRA cho Qwen3.5 vì lỗi lượng tử hoá cao hơn bình thường. Cần đo VRAM và chất
lượng thực tế để xác nhận._

---

## 5. Phán quyết (NB5)

> ⚠ **Cần GPU để chạy NB5.** Phần này chưa chạy được trên máy CPU-only.

**Kết quả cổng hồi quy**: _chưa chạy_
`target Δ = _chưa đo_` · `regression Δ = _chưa đo_` · `valid_trace_rate = _chưa đo_`

Diễn giải (≥100 từ). Nếu FAILED: **vì sao**, và điều đó nói gì về bài toán của bạn?
(Một FAILED được phân tích tốt ăn điểm cao hơn một PASSED không giải thích được.)

_Chưa viết được — cần kết quả từ NB5. Tuy nhiên, dựa trên thiết kế của lab, nếu fine-tune
không thắng được baseline (b) — prompt đã tối ưu — thì kết luận đúng là "bài toán này
không cần fine-tune". Đó là một kết quả hợp lệ và được chấm điểm đầy đủ._

---

## 6. Định tính — bắt buộc có cả ca THUA

> ⚠ **Cần GPU để chạy NB5.** Phần này chưa chạy được trên máy CPU-only.

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | _chưa đo_ | | | | ✅ FT thắng |
| 2 | _chưa đo_ | | | | ✅ FT thắng |
| 3 | _chưa đo_ | | | | ❌ **FT thua** |
| 4 | _chưa đo_ | | | | ❌ **FT thua** |
| 5 | _chưa đo_ | | | | |

Có mẫu chung nào ở các ca FT thua không?

_Chưa phân tích được — cần kết quả từ NB5._

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** Bạn có nên deploy bản fine-tune này không, và vì sao? Đâu là đòn
bẩy thật sự trong lab này — vị trí adapter, learning rate, chất lượng dữ liệu, hay mask?

Lab Day 21 này dạy một bài học quan trọng: **fine-tuning không phải là câu trả lời mặc định cho mọi bài toán**. Điểm mấu chốt nằm ở việc thiết kế một phép so sánh công bằng — đo baseline (b) với prompt đã tối ưu **trước khi** train, rồi mới quyết định xem fine-tune có thật sự thắng hay không. Từ NB1, tôi đã học được rằng loss mask và chat template quyết định kết quả nhiều hơn mọi biến thể LoRA cộng lại. Mask đúng nghĩa là chỉ tính loss trên câu trả lời, không tính trên câu hỏi — nếu tính cả prompt, model sẽ học cách viết lại câu hỏi thay vì trả lời. Về đòn bẩy thật sự, deck §10 chỉ ra rằng vị trí gắn adapter (text-linear vs attn-only) và learning rate (10x full-FT) quan trọng hơn rank. Rank chỉ là "năng lực so với lượng thông tin trong dữ liệu" — với 250 mẫu, rank cao không tự động tốt hơn. Cuối cùng, việc phát hiện ra rằng một fine-tune có thể thua baseline (b) là một kết quả hợp lệ — nó cho thấy prompt engineering đôi khi đủ tốt và không cần fine-tune. Quyết định deploy phụ thuộc vào kết quả NB5: nếu fine-tune thắng (b) trên target mà không tụt regression, thì deploy; nếu không, giữ nguyên base + prompt tối ưu.

**Ba điều tôi học được** (cụ thể, không generic):
1. **Loss mask là nền tảng của mọi thứ.** Nếu mask sai (tính loss cả prompt), model sẽ học viết lại câu hỏi thay vì trả lời — và không có lỗi nào báo hiệu. NB1 bắt tôi phải **đọc** đoạn được tính loss, không chỉ tin vào cờ của thư viện.
2. **So sánh công bằng cần khớp ngân sách tham số, không khớp rank.** So `q,v @ r=16` với `all-linear @ r=16` là so ngân sách, không phải so vị trí. `matched_rank()` giải ra rank đưa attention-only về đúng ngân sách của `correct` — chỉ khi đó biến duy nhất còn lại mới là vị trí.
3. **LR thang full-FT không chuyển được sang LoRA.** LoRA cần LR cao hơn ~10x (1e-4 thay vì 1e-5). Nếu dùng sai thang, loss gần như phẳng từ step 0 — và nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai rằng "LoRA không học được".

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Chạy NB2-NB5 trên Colab Free T4 để có số đo thực tế, đặc biệt là so sánh thứ tự xếp hạng giữa `final_loss` (NB4) và điểm target (NB5 §4) — nếu hai thứ tự khác nhau, đó là bằng chứng trực tiếp cho Lỗi #3 mà lab cảnh báo.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link: