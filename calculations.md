# Calculations - LAB 10

## 1. Calculation Exercise 1 - Start / End

Context tokens (index từ 0):

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Token | The | capital | of | France | is | Paris | . |

Answer "Paris": start = 5, end = 5.

Answer "capital of France": start = 1, end = 3.

Kiểm tra: token 1 đến token 3 = [capital, of, France], thỏa i ≤ j.

## 2. Calculation Exercise 2 - Start Loss

Gold start = 5, P_start(5) = 0.60 (tổng xác suất = 0.02 + 0.05 + 0.03 + 0.10 + 0.15 + 0.60 + 0.05 = 1.00).

L_start = -ln(0.60) = 0.5108

End probability: đề không cho bảng nên giả sử gold end = 5 (answer "Paris") và model dự đoán:

| Position | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| P_end | 0.01 | 0.02 | 0.02 | 0.05 | 0.10 | 0.70 | 0.10 |

L_end = -ln(0.70) = 0.3567

## 3. QA Loss

L = (L_start + L_end) / 2 = (0.5108 + 0.3567) / 2 = 0.4338

So sánh: nếu model chỉ cho P_start(5) = 0.15 thì L_start = -ln(0.15) = 1.8971, loss tăng gần 4 lần. Nếu P_start(5) = 1 thì L_start = 0.

Vì sao cần hai prediction thay vì một:

- Answer là một span, được xác định bởi hai vị trí (i, j). Một classifier trên T vị trí chỉ chọn được một token nên chỉ biểu diễn được answer dài 1 token.
- Nếu coi mỗi span là một class thì có T(T+1)/2 class (T = 384 thì có 73.920 class), rất tốn và khó học. Tách thành start classifier và end classifier chỉ cần 2 x T output, P(span i:j) = P_start(i) x P_end(j).
- Start và end mang thông tin khác nhau (ranh giới trái và ranh giới phải), mỗi head học riêng một ranh giới, lỗi cũng có thể phân tích riêng (wrong start / wrong end).
- Token classification kiểu BIO không đảm bảo output là đúng một span liên tục, còn start/end + ràng buộc i ≤ j, độ dài ≤ max_answer_len luôn cho đúng một span.

## 4. Tokenization của QA (Section 10)

Input: question = "What is the capital of France?", context = "The capital of France is Paris.", tokenizer distilbert-base-uncased (giống bert-base-uncased).

| Idx | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| token | [CLS] | what | is | the | capital | of | france | ? | [SEP] | the | capital | of | france | is | paris | . | [SEP] |
| input_ids | 101 | 2054 | 2003 | 1996 | 3007 | 1997 | 2605 | 1029 | 102 | 1996 | 3007 | 1997 | 2605 | 2003 | 3000 | 1012 | 102 |
| attention_mask | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| token_type_ids | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

- BERT dùng token_type_ids (segment A = 0 cho [CLS] + question + [SEP] đầu, segment B = 1 cho context + [SEP] cuối).
- DistilBERT: tokenizer vẫn trả về token_type_ids nhưng model không có segment embedding, forward() không nhận tham số này nên phải bỏ đi khi đưa vào model.
- RoBERTa: tokenizer không trả về token_type_ids, format là `<s> question </s></s> context </s>`.
- Do đó phân biệt question/context bằng sequence_ids() của tokenizer (0 = question, 1 = context, None = special token), không dựa vào token_type_ids.

## 5. Calculation Exercise 3 - Character Span

Context: "The capital of France is Paris." (31 ký tự, index 0 - 30)

| Char idx | 25 | 26 | 27 | 28 | 29 | 30 |
|---|---|---|---|---|---|---|
| Ký tự | P | a | r | i | s | . |

1. Index ký tự đầu tiên của Paris: 25 (= answer_start).
2. Index ký tự cuối: 29.
3. Answer span theo character: [25, 29] (inclusive), hay [25, 30) theo kiểu Python, answer_end = 25 + len("Paris") = 30, context[25:30] = "Paris".
4. Sau tokenization: token "paris", offset (25, 30). Đây là token index 14 trong input [CLS] question [SEP] context [SEP] (token index 5 nếu chỉ tính context, word index 5).
5. Start token = 14 vì offset start 25 ≤ answer_start 25 < 30.
6. End token = 14 vì 25 < answer_end 30 ≤ 30.

Ba loại index cho cùng "Paris": character index 25, word index 5, token index 14 (trong input QA). Dùng 25 làm label cho model là sai.

## 6. Token-level F1 (tự tính)

Ví dụ 1: Gold "Paris France", Prediction "France".

- overlap = {france} = 1, #predicted = 1, #gold = 2
- P = 1/1 = 1.0, R = 1/2 = 0.5
- F1 = 2 x 1.0 x 0.5 / (1.0 + 0.5) = 0.667, EM = 0

Ví dụ 2: Gold "New York City", Prediction "York City".

- overlap = 2, #predicted = 2, #gold = 3
- P = 2/2 = 1.0, R = 2/3 = 0.667
- F1 = 2 x 1.0 x 0.667 / 1.667 = 0.8, EM = 0

Ví dụ 3: Gold "Paris", Prediction "the city of Paris".

- Sau normalization (lowercase, bỏ dấu câu, bỏ a/an/the): pred = [city, of, paris], gold = [paris]
- overlap = 1, P = 1/3 = 0.333, R = 1/1 = 1.0
- F1 = 2 x 0.333 x 1.0 / 1.333 = 0.5, EM = 0

Ví dụ 4: Gold "Paris", Prediction "paris." -> sau normalization đều là "paris", EM = 1, F1 = 1.
