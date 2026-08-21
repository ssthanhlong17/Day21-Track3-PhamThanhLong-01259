# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Điều ngạc nhiên nhất là loss mask và chat template quyết định kết quả nhiều hơn mọi biến thể LoRA cộng lại. Tôi tưởng rằng việc chọn rank, learning rate, hay số lớp adapter mới là yếu tố quyết định — nhưng NB1 cho thấy nếu mask sai (tính loss cả prompt), model sẽ học cách viết lại câu hỏi thay vì trả lời, và không có lỗi nào báo hiệu. Còn nữa, tôi ngạc nhiên khi biết rằng `all-linear` của PEFT trên model đa phương thức sẽ gắn adapter vào cả vision tower — một chi tiết kỹ thuật nhỏ nhưng có thể làm hỏng toàn bộ checkpoint.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Tôi mất nhiều thời gian nhất ở việc hiểu cách mask hoạt động — cụ thể là tại sao diff token lists lại sai trên Qwen3.5. Lý do là generation prompt kết thúc bằng ` thinking\n` trong khi full render tiếp tục `\n response`, và hai newline liên tiếp gộp thành một token khác. Vì vậy, hai danh sách token không phải là prefix của nhau dù chuỗi ký tự thì có. Đây không phải chỗ tôi dự đoán — tôi tưởng diff token lists là cách đơn giản và đúng đắn.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi từng tin rằng "fine-tuning luôn tốt hơn prompt engineering" và rằng rank cao hơn luôn tốt hơn. Lab này cho thấy cả hai đều sai: một prompt được tối ưu tốt (baseline b) có thể thắng cả fine-tune, và rank chỉ là "năng lực so với lượng thông tin trong dữ liệu" — với 250 mẫu, rank cao không tự động tốt hơn. Tôi cũng từng tin rằng loss thấp = model tốt, nhưng lab chỉ ra rằng loss thấp có thể chỉ là ghi nhớ, và phải chấm bằng điểm trên tập target.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng AI assistant để giải thích các khái niệm LoRA/QLoRA, giúp đọc hiểu code trong `labkit/`, và hỗ trợ viết report. Nó sai ở chỗ ban đầu đề xuất dùng `target_modules="all-linear"` trực tiếp — không biết rằng trên Qwen3.5 (model đa phương thức), điều này sẽ gắn adapter vào cả vision tower. Phải đọc kỹ `modeling.py` mới phát hiện ra vấn đề này.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Bước đầu tiên là **đo baseline trước khi train** — chạy model base với prompt đã tối ưu trên tập eval đã đóng băng, để biết mốc phải vượt là gì. Nếu prompt engineering đã đạt kết quả tốt, có thể không cần fine-tune. Đồng thời, tôi sẽ kiểm tra loss mask bằng cách **đọc** đoạn được tính loss (không chỉ tin vào cờ của thư viện), và đảm bảo prompt dùng lúc train giống hệt prompt dùng lúc eval — vì F-31 đã chứng minh rằng lệch prompt giữa train và eval làm mọi adapter đạt 0.000.