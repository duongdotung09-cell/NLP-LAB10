# LAB 10 - Extractive Question Answering

Fine-tune DistilBERT cho extractive QA trên SQuAD v1.1: (question, context) -> (start, end) -> answer span. Toàn bộ pipeline tự cài đặt (tokenize với sliding window, offset mapping, căn chỉnh span, training loop, giải mã span, EM/F1), không dùng QA pipeline có sẵn.

## Cấu trúc

| File                       | Nội dung                                                                                                           |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| README.md                  | Tổng quan, cách chạy, reproducibility, kết quả chính                                                          |
| calculations.md            | Exercise 1 - 3, QA loss, tokenization input, ví dụ tính F1                                                       |
| prediction.md              | Prediction trước khi chạy và đối chiếu với kết quả                                                        |
| tokenizer_alignment.ipynb  | input_ids / attention_mask / token_type_ids, offset mapping, sliding window, alignment và các trường hợp biên |
| qa_model.py                | Module lõi: alignment, feature, training, post-processing, metric, baseline, run_experiment                        |
| qa_finetuning.ipynb        | Baseline, Experiment 1 - 5, kiểm tra Prediction 3 - 4, thống kê lỗi, sinh hình và results.csv                 |
| results.csv                | Kết quả tất cả experiment                                                                                       |
| learning_curves.png        | Training / validation loss                                                                                          |
| em_f1_analysis.png         | EM vs F1 của baseline và model, phân bố EM = 1 / F1 > 0 / F1 = 0                                                |
| answer_length_analysis.png | EM / F1 theo độ dài answer                                                                                       |
| error_analysis.md          | Phân loại lỗi, wrong start vs wrong end, ví dụ EM = 0 nhưng F1 > 0                                            |
| document_qa.py             | Mini Document QA trên tài liệu tự chọn                                                                         |
| reflection.md              | Reflection LAB 08 - 10, QA vs RAG, no-answer, Individual Learning Check                                             |
| outputs/                   | Log, history, prediction trên test, model đã fine-tune (outputs/len384/model)                                    |

## Cách chạy

```
pip install torch transformers datasets pandas matplotlib jupyter
python qa_model.py 384 128 1      # fine-tune model chính, lưu model vào outputs/len384/model
python qa_model.py 128 42         # Experiment 1
python qa_model.py 256 85         # Experiment 1
jupyter nbconvert --to notebook --execute --inplace tokenizer_alignment.ipynb
jupyter nbconvert --to notebook --execute --inplace qa_finetuning.ipynb
python document_qa.py             # demo; -i để hỏi tương tác, -f file.txt để dùng tài liệu khác
```

Notebook qa_finetuning.ipynb tự gọi run_experiment nếu outputs/lenXXX chưa có kết quả, nên có thể chạy notebook trực tiếp mà không cần các lệnh train ở trên. Trên CPU mỗi cấu hình mất khoảng 1 giờ.

## Reproducibility

```
Model: distilbert-base-uncased + linear QA head (768 -> 2)
Tokenizer: distilbert-base-uncased (WordPiece, fast tokenizer)
Dataset: SQuAD v1.1 (rajpurkar/squad trên Hugging Face)
Train split: 3.000 câu ngẫu nhiên từ SQuAD train (shuffle seed 42) -> 3.030 feature
Validation split: 300 câu từ SQuAD validation (index 1000 - 1299 sau shuffle seed 42), dùng cho validation loss
Test split: 1.000 câu từ SQuAD validation (index 0 - 999 sau shuffle seed 42), dùng cho EM/F1
Max sequence length: 384 (Experiment 1: 128 / 256 / 384)
Stride: 128 (Experiment 1: 42 / 85 / 128, overlap = 1/3 max_length)
Batch size: 16, dynamic padding
Learning rate: 5e-5, warmup 10%, linear decay, weight decay 0.01, gradient clipping 1.0
Epochs: 2
Optimizer: AdamW
Random seed: 42
Hardware: Intel Core i9-13900H (14 threads), 16 GB RAM, không GPU, torch 2.12.0+cpu, transformers 5.14.1
Training time: 70.5 phút (384), 60.3 phút (128), 52.9 phút (256)
Post-processing: n_best = 20, max_answer_len = 30 token, chọn span có start_logit + end_logit lớn nhất trên mọi window
```

Do giới hạn CPU nên chỉ train trên 3.000 câu (khoảng 3.4% SQuAD train). DistilBERT train trên toàn bộ SQuAD đạt khoảng 77 EM / 85 F1, nên kết quả ở đây thấp hơn nhiều. Training time giữa các cấu hình không so sánh chính xác được: lần 384 chạy song song với notebook khác, còn lần 128 bị chậm khi máy rút sạc.

## Kết quả chính (1.000 câu test)

| Model                       | EM   | F1   |
| --------------------------- | ---- | ---- |
| Baseline keyword overlap    | 11.2 | 20.6 |
| Fine-tuned DistilBERT (384) | 46.7 | 57.5 |

Learning curves: validation loss giảm từ 4.05 xuống 2.02 sau 2 epoch. Từ epoch 1.25 training loss tiếp tục giảm còn validation loss gần như đi ngang, model bắt đầu overfit nhẹ trên 3.000 câu train.

Experiment 1 - context length:

| Max length | EM   | F1   | Training time (phút) |
| ---------- | ---- | ---- | --------------------- |
| 128        | 46.4 | 57.6 | 60.3                  |
| 256        | 45.2 | 56.5 | 52.9                  |
| 384        | 46.7 | 57.5 | 70.5                  |

Tăng context length không cải thiện QA ở đây. Sliding window với stride đã đảm bảo mọi answer nằm trọn trong ít nhất một window, và đáp án SQuAD thường chỉ cần ngữ cảnh cục bộ. Chênh lệch khoảng 1 điểm nằm trong nhiễu.

Experiment 4 - lexical overlap (F1): high overlap 65.5, medium 61.1, low overlap 45.8. Model phụ thuộc mạnh vào việc question trùng từ với câu chứa đáp án, nhưng ở nhóm low overlap vẫn hơn baseline 4 lần (45.8 so với 11.4).

Experiment 5 - answer length (EM / F1), độ dài = số từ (tách theo khoảng trắng) của gold answer đầu tiên, dùng đơn vị từ thay cho subword token để khớp với cách EM/F1 tách token: 1 từ 51.2 / 57.4, 2 - 3 từ 52.3 / 62.9, 4 - 6 từ 38.7 / 54.7, 7+ từ 20.2 / 37.1.

Error analysis: 533 câu EM = 0, trong đó 64% chọn sai vị trí hoàn toàn (wrong span), 15.6% chỉ sai start, 14.8% chỉ sai end, 5.6% sai cả hai đầu. 236 câu có EM = 0 nhưng F1 > 0.

## Document QA

document_qa.py dùng model outputs/len384/model trên một tài liệu 479 token (2.341 ký tự) về Transformer, BERT, DistilBERT, SQuAD. Tài liệu dài hơn 384 token nên được chia 2 window. Mỗi câu trả lời in kèm score (start_logit + end_logit), vị trí ký tự và câu chứa đáp án làm evidence.

Kết quả demo trên 13 câu hỏi có đáp án trong tài liệu:

- Đúng 4 câu: "15 percent", "536", "semantic retrieval", "Hugging Face".
- Đúng một phần 2 câu: "24 layers and about 340 million" (thừa "24 layers and"), định nghĩa Exact Match (thừa "The F1 score").
- "2018" cho câu hỏi năm phát hành BERT đúng về chữ nhưng lấy từ câu về SQuAD 2.0 (distractor cùng năm).
- Sai 6 câu, đa số là distractor cùng loại thực thể: "110 million" cho câu hỏi số layer, "97 percent" thay vì "60 percent", "82.3" (EM) thay vì 91.2 (F1), "2018" cho hạn nộp Lab 10.

Câu hỏi không có đáp án trong tài liệu ("Who is the president of France?") vẫn nhận một span vì model train trên SQuAD 1.1, nhưng score thấp (1.32) nên bị gắn cảnh báo độ tin cậy thấp (ngưỡng mặc định 5.0). Ngưỡng này chưa tách được hết câu sai: nhiều câu sai vẫn có score 9 - 11.

## AI Usage

- AI-assisted code: Claude Code
- What was generated:
  - qa_model.py: tải và chia SQuAD theo seed, collate với dynamic padding, vòng lặp train/evaluate (warmup, linear decay), lưu kết quả vào outputs/lenXXX và results.csv.
  - outputs/make_ft_nb.py: phần khung nbformat (hàm md()/code(), ghi file .ipynb) để ghép các cell thành qa_finetuning.ipynb. Nội dung các cell phân tích (phân loại lỗi, chia nhóm Exp 3 - 5, perturbation Prediction 3 - 4) do tự viết, không do AI sinh.
  - Code vẽ 3 hình (learning_curves.png, em_f1_analysis.png, answer_length_analysis.png).
  - document_qa.py: phần đọc tham số dòng lệnh và in kết quả.
  - Định dạng bảng trong các file .md (không viết nội dung).
- What was modified (tự làm):
  - Tự viết char_span_to_token_span (kể cả nhánh xử lý answer bắt đầu/kết thúc ở khoảng trắng), prepare_features, postprocess, EM/F1 và baseline; AI chỉ giải thích error message khi debug, không viết logic.
  - Tự định nghĩa quy tắc phân loại lỗi (so vị trí ký tự start/end sau khi bỏ dấu câu hai đầu: wrong start / wrong end / wrong boundary / wrong span), cách chia nhóm Exp 4 (tercile lexical overlap) và Exp 5 (số từ của gold answer), cách tạo perturbation cho Prediction 3 - 4.
  - Tự chọn cấu hình (DistilBERT, 3.000 câu train, 2 epoch, stride = 1/3 max_length) và tự soạn tài liệu, câu hỏi demo cho document_qa.py.
  - Tự viết calculations.md, prediction.md, error_analysis.md, reflection.md và tự cập nhật số liệu theo output. Phần Prediction không sửa sau khi chạy, phần Đối chiếu viết sau.
- What was verified:
  - Alignment trên 3.000 câu train với cả 3 max_length (khôi phục đúng 3000/3008 feature ở 384), đọc từng trường hợp lệch.
  - Hàm EM/F1 trên các cặp nhỏ (mạo từ, dấu câu, thừa/thiếu từ, nhiều gold); baseline và model chấm trên cùng 1.000 câu test.
  - Đọc lại question, context, prediction của từng lỗi trong error_analysis.md và đối chiếu với nhãn do code gán.
  - Đối chiếu số liệu trong các file .md với results.csv và log.
