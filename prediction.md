# Prediction Before Experiment

Ghi trước khi chạy fine-tuning (DistilBERT, 3.000 câu hỏi SQuAD train, test trên 1.000 câu SQuAD validation).

## Prediction 1 - Nhiều từ trùng giữa question và context

Có. Model fine-tune ít dữ liệu dễ học heuristic "chọn span nằm gần các từ trùng với question". Nếu nhiều vị trí trong context cùng chứa từ của question thì model dễ chọn nhầm span gần một vị trí sai, nhất là khi các span cùng kiểu thực thể (cùng là năm, cùng là tên người). Dự đoán: câu hỏi có lexical overlap cao với câu chứa đáp án sẽ có EM/F1 cao hơn rõ rệt so với câu hỏi diễn đạt khác (paraphrase), chênh lệch khoảng 10 - 20 điểm F1.

## Prediction 2 - Answer span dài hơn

Có, khó hơn. Span dài thì phải đúng cả hai ranh giới cách xa nhau, ranh giới cuối thường mơ hồ (có lấy mệnh đề phụ, dấu câu, giới từ hay không). Dự đoán: EM giảm mạnh khi answer dài (1 token cao nhất, 7+ token thấp nhất, có thể dưới 30%). F1 giảm ít hơn EM vì span dài vẫn được điểm overlap một phần. Lỗi wrong end sẽ nhiều hơn wrong start.

## Prediction 3 - Context có nhiều thông tin không liên quan / distractor

Performance giảm. Thông tin không liên quan đơn thuần ít ảnh hưởng vì attention vẫn tìm được câu liên quan, nhưng distractor (câu có từ giống question và có thực thể cùng loại với đáp án) làm model chọn nhầm span. Context dài hơn còn có thể bị chia thành nhiều window, đáp án ở window khác và score giữa các window không so sánh tốt. Dự đoán: lỗi distractor và multiple plausible spans chiếm phần lớn lỗi wrong span.

## Prediction 4 - Đổi thứ tự câu trong context

Ảnh hưởng nhỏ. Transformer có position embedding nhưng câu trả lời chủ yếu phụ thuộc quan hệ question - câu chứa đáp án, nên đa số dự đoán không đổi. Một số câu sẽ đổi kết quả khi câu chứa đáp án dựa vào câu trước đó (đại từ "it", "he", "this") hoặc khi đáp án bị đẩy sang window khác do truncation.

## Dự đoán cho Experiment 1 - Context length

Tăng max_length từ 128 lên 256 sẽ cải thiện rõ (ít window, model thấy ngữ cảnh rộng hơn, ít feature chỉ có nhãn [CLS]). Từ 256 lên 384 cải thiện ít vì phần lớn context SQuAD ngắn hơn 256 token. Training time tính theo số example gần như không tăng nhiều do dùng dynamic padding.

## Đối chiếu sau khi chạy

Phần này viết sau khi có kết quả (qa_finetuning.ipynb, results.csv). Phần prediction ở trên giữ nguyên.

| Prediction | Kết quả thực tế | Đúng / sai |
|---|---|---|
| 1. Overlap cao dễ hơn, chênh 10 - 20 F1 | F1 high overlap 65.5, low overlap 45.8 (chênh 19.7), EM 53.9 so với 34.8 | Đúng |
| 2. Span dài khó hơn, 7+ token EM < 30% | EM: 1 từ 51.2, 2-3 từ 52.3, 4-6 từ 38.7, 7+ từ 20.2. F1: 57.4, 62.9, 54.7, 37.1 | Đúng phần lớn |
| 2. Wrong end nhiều hơn wrong start | Chỉ sai end 79, chỉ sai start 83 | Sai |
| 3. Distractor làm giảm performance | Chèn 3 câu lạ trước câu chứa answer (300 câu): EM 47.3 -> 42.7, F1 58.6 -> 54.6 | Đúng |
| 4. Đổi thứ tự câu ảnh hưởng nhỏ | EM 47.3 -> 44.3, F1 58.6 -> 57.2, nhưng 113/300 câu đổi prediction | Đúng về metric, sai về độ ổn định |
| Exp 1. 128 -> 256 cải thiện rõ | F1: 128 là 57.6, 256 là 56.5, 384 là 57.5 | Sai |

Nhận xét:

- Prediction 1: model phụ thuộc rõ vào lexical overlap, F1 giảm gần 20 điểm khi câu hỏi diễn đạt khác câu chứa đáp án. Các cặp câu thủ công (Paris, Shakespeare, 1945) đều đúng kể cả khi paraphrase, vì context chỉ có một câu và một ứng viên duy nhất. Khó khăn chỉ xuất hiện khi context có nhiều ứng viên cạnh tranh.
- Prediction 2: độ dài đo bằng số từ của gold answer, không phải số subword token. Span dài khó hơn đúng như dự đoán, EM nhóm 7+ chỉ 20.2. Tuy nhiên nhóm 1 từ không cao nhất: đáp án 1 từ thường là năm, số hoặc tên, và context SQuAD hay có nhiều năm/số cùng loại (distractor). F1 giảm chậm hơn EM đúng như dự đoán (nhóm 7+ có F1 37.1 so với EM 20.2). Lỗi ranh giới chia đều cho start và end, không lệch về end như dự đoán.
- Prediction 3: distractor làm F1 giảm 4 điểm. Mức giảm vừa phải vì câu chèn vào lấy ngẫu nhiên từ đoạn khác, ít khi cùng chủ đề với question. Distractor thật trong cùng đoạn văn gây hại nhiều hơn: 64% lỗi của model là wrong span (error_analysis.md).
- Prediction 4: metric gần như không đổi (F1 giảm 1.4), nhưng hơn 1/3 câu đổi prediction. Vì EM chỉ giảm 3 điểm, phần lớn các thay đổi này là chuyển từ một span sai sang một span sai khác, hoặc chỉ lệch ranh giới. Như vậy model kém ổn định ở những câu nó không chắc chắn, dù metric trung bình gần như giữ nguyên.
- Experiment 1: dự đoán sai. Với stride đủ lớn, cả 3 cấu hình đều có window chứa trọn answer cho 3000/3000 câu train. Đáp án SQuAD thường xác định được trong ngữ cảnh cục bộ (một hai câu quanh đáp án), nên window 128 token vẫn đủ. Cấu hình 128 còn sinh nhiều feature hơn (6425 so với 3030), tức gấp đôi số bước cập nhật và có thêm ví dụ âm nhãn [CLS]. Chênh lệch giữa 3 cấu hình khoảng 1 điểm, nằm trong sai số của tập test 1000 câu. Tăng context length không luôn cải thiện QA, chỉ có lợi khi đáp án cần ngữ cảnh xa.
