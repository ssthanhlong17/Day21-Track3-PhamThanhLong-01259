# Lab 21 — Evaluation Report

**Họ tên**: Phạm Thanh Long  **MSSV**: 01259  **Ngày**: 2026-08-21
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: Tesla T4 (14.6 GB)

> Các số liệu được đối chiếu với `results/`.

## 1. Setup

| Hạng mục | Giá trị |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage |
| Train / val | 225 / 25, seed 42 |
| `max_length` | 1024; p95 là 98, gợi ý tối thiểu 256 |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 / 30 |

Template giữ reasoning/thinking khi render (`results/template_check.json`), nên trace trong assistant turn có thể tới loss.

## 2. Mask proof (NB1)

| Chỉ số | Giá trị |
|---|---|
| `supervised_fraction` | 0.4149 |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đầu phần được supervised:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

## 3. Ba baseline (NB2)

| Run | target | regression | format | latency (ms) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.000 | 0.7578 | 0.000 | 3179.8 |
| (b) base + optimized prompt | 0.765 | 0.7578 | 1.000 | 1008.3 |
| (c) LoRA fine-tune | 0.975 | 0.5444 | 1.000 | 1408.5 |

Baseline (b) mạnh hơn rất rõ (a): target tăng 0.765, format từ 0 lên 1.0 và latency thấp hơn. Tôi không sửa `OPTIMIZED_PROMPT`; SHA `719e74d3b6232053` khớp prompt của lab. Run này dùng đủ 50 target và 15 regression, `eval_limit` là `null`.

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss | target | VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6250 | **0.975** | 12.01 |
| `attn_only` | q,v | 283 | 32,456,704 | 1e-4 | **0.5371** | 0.970 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5704 | 0.000 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.940 | **7.09** |

**4.1.** `attn_only` dùng 32,456,704 tham số, gần bằng `correct` 32,464,896 (chênh khoảng 0.025%). Nó thua nhẹ trên target, 0.970 so với 0.975, dù lại có train loss thấp hơn, 0.5371 so với 0.6250. Thứ tự theo loss vì thế đảo ngược thứ tự theo tác vụ. Rank đã tăng đến 283 để khớp ngân sách nhưng không thay thế được placement text-linear rộng hơn.

**4.2.** `wrong_lr` chỉ thay LR từ 1e-4 xuống 1e-5 nhưng final loss tăng lên 1.5704 và target/format rơi về 0.000. Nếu không biết LR, có thể kết luận sai rằng LoRA hoặc dữ liệu không học được. Đây là sai thang learning rate của LoRA, không phải bằng chứng placement sai.

**4.3.** QLoRA dùng 7.09 GB so với 12.01 GB, tiết kiệm 4.92 GB, xấp xỉ 41%. Đánh đổi là target giảm 0.975 → 0.940 và latency tăng 1408.5 → 1810.7 ms. T4 vẫn chứa fp16, nên số đo ủng hộ việc không chọn QLoRA mặc định cho model này; chỉ dùng khi VRAM là ràng buộc cứng.

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy: FAILED**
`target Δ = +0.210` · `regression Δ = -0.2133` · `valid_trace_rate = 0.0`

Fine-tune vượt baseline prompt tối ưu ở target (0.975 so với 0.765), nhưng regression giảm từ 0.7578 xuống 0.5444. Mức giảm 0.2133 lớn hơn rất nhiều ngưỡng 0.020, vì vậy FAILED là kết luận đúng, không nên nới gate để lấy PASS. Đây là catastrophic forgetting: 225 mẫu triage đã kéo adapter theo miền mới nhưng làm giảm năng lực tổng quát. Tôi không deploy adapter này. Hướng sửa là thêm 1–5% replay data tổng quát, giảm cường độ fine-tune hoặc dừng sớm, rồi chạy lại chính regression gate này.

## 6. Định tính — có ca FT thua

`qualitative.json` lưu score/dự đoán của FT; NB2 không lưu prediction theo từng item của prompt (b), nên bảng không bịa prediction của (b), chỉ dùng target tổng 0.765.

| # | Ticket rút gọn | Nhãn đúng | (b) prompt | (c) FT | Nhận xét |
|---|---|---|---|---:|---|
| 1 | Chuột không dây, trả lại gấp | `doi_tra`, cao, tích cực | target tổng 0.765 | 1.00 | FT thắng |
| 2 | Ốp lưng, hoàn tiền sớm | `hoan_tien`, trung bình, tiêu cực | target tổng 0.765 | 1.00 | FT thắng |
| 3 | Bình giữ nhiệt, chưa thấy tiền | `hoan_tien`, thấp, tích cực | target tổng 0.765 | **0.75** | **FT thua**: sai urgency |
| 4 | Áo khoác gió bị lỗi, khi nào tiện | `san_pham_loi`, thấp, tích cực | target tổng 0.765 | **0.75** | **FT thua**: sai urgency |
| 5 | Đèn bàn LED giao chậm, khi nào tiện | `van_chuyen`, thấp, tích cực | target tổng 0.765 | **0.75** | **FT thua**: sai urgency |

Các ca FT thua đều có cụm “Khi nào tiện”: nhãn đúng là urgency `thap`, nhưng model dự đoán `trung_binh`. Intent, product và sentiment vẫn đúng nên score là 0.75.

## 7. Kết luận & điều tôi học được

Tôi không deploy fine-tune này. Nó tăng target từ 0.765 của prompt tối ưu lên 0.975, nhưng giảm regression 0.2133 và FAILED gate. Với CSKH, lợi ích ở miền mới không bù cho suy giảm năng lực tổng quát nếu không có replay hoặc routing. Kết quả cho thấy rank không phải đòn bẩy duy nhất: với ngân sách gần bằng nhau, text-linear thắng attention-only trên target dù attention-only có train loss thấp hơn. Learning rate có tác động rất lớn; giảm 10 lần đã làm target và format về 0. QLoRA tiết kiệm 41% VRAM nhưng đổi bằng chất lượng và latency. Mask đúng là điều kiện nền tảng: assistant-only xác nhận prompt bị mask và JSON đáp án thực sự được supervised. Nếu có thêm thời gian, tôi sẽ thêm replay data và giảm epoch/LR, sau đó chạy lại cùng 50 target + 15 regression thay vì chọn checkpoint theo train loss.

**Ba điều tôi học được:**

1. Train loss có thể xếp hạng sai deployment quality: 0.5371 của `attn_only` tốt hơn 0.6250 của `correct`, nhưng target lại thấp hơn.
2. So sánh placement phải khớp ngân sách: 32,456,704 vs 32,464,896 trainable params là một contrast công bằng.
3. Regression gate bảo vệ khỏi kết luận hấp tấp: target tăng 0.210 nhưng regression giảm 0.2133, nên không deploy.

## Phụ lục — thưởng

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng
- [ ] B3 reasoning-trace collapse
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
