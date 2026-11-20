# Reflection - LAB 10

## 1. LAB 08, LAB 09, LAB 10: cùng backbone, khác prediction head

```
                     Pretrained Transformer
                              │
            ┌─────────────────┼──────────────────┐
            ↓                 ↓                  ↓
   [CLS] representation   Token reps         Token reps
     h_CLS (1 x d)        H (T x d)          H (T x d)
            ↓                 ↓                  ↓
   Linear d -> K       Linear d -> K/token   Linear d -> 2/token
            ↓                 ↓                  ↓
     Classification          NER            Start / End
   softmax trên K lớp   softmax trên K nhãn  softmax trên T vị trí
                          cho từng token      (2 lần: start, end)
            ↓                 ↓                  ↓
         LAB 08            LAB 09             LAB 10
```

Cả ba lab dùng cùng một pretrained Transformer. Backbone nhận chuỗi token và trả về ma trận H gồm T vector ngữ cảnh, mỗi vector d chiều. Phần khác nhau chỉ nằm ở hai chỗ: lấy phần nào của H và target là gì.

LAB 08 (sequence classification) cần một nhãn cho cả văn bản. Ta chỉ lấy vector h_[CLS], cho qua một lớp Linear d -> K rồi softmax trên K lớp. Target là một số nguyên cho mỗi văn bản, loss là cross-entropy trên K lớp. Các token còn lại không được dùng trực tiếp, thông tin của chúng đi vào [CLS] qua self-attention.

LAB 09 (token classification, NER) cần một nhãn cho mỗi token. Ta dùng toàn bộ H, cùng một lớp Linear d -> K áp cho từng token. Target là chuỗi nhãn BIO dài T, loss là trung bình cross-entropy trên các token (subword phụ và special token gán -100). Softmax ở đây vẫn chạy trên trục nhãn.

LAB 10 (extractive QA) cần một đoạn trong context. Input là cặp [CLS] question [SEP] context [SEP], nên self-attention cho phép mỗi token context "nhìn" question. Head là Linear d -> 2 áp lên từng token, tạo ra start logit và end logit. Điểm khác quan trọng nhất là softmax chạy trên trục vị trí (T token) chứ không trên trục nhãn: mỗi example chỉ có hai target là chỉ số token i và j. Loss = (CE_start + CE_end) / 2. Khi suy luận phải giải mã span: chọn i ≤ j, giới hạn độ dài, chỉ xét token thuộc context, gom nhiều window của một context dài, rồi dùng offset mapping đổi về đoạn ký tự.

Vì sao cùng backbone mà head và target khác nhau? Pretraining (masked language modeling) dạy backbone tạo biểu diễn ngữ cảnh tốt cho mọi token, nhưng không gắn với một bài toán cụ thể nào. Dạng output của bài toán quyết định head: một nhãn cho cả câu thì cần một vector tổng hợp, một nhãn mỗi token thì cần classifier theo token, một đoạn văn bản thì cần phân phối trên vị trí. Target cũng thay đổi theo: class id, chuỗi nhãn, cặp (start, end). Head mới chỉ có rất ít tham số (với QA chỉ có 768 x 2 + 2 tham số) và khởi tạo ngẫu nhiên, còn fine-tuning cập nhật cả backbone để biểu diễn phù hợp với head đó.

Thực nghiệm của LAB 10 cho thấy rõ điều này. Cùng DistilBERT nhưng muốn làm QA thì phải xử lý thêm những thứ classification không cần: offset mapping, sliding window với stride, nhãn [CLS] cho window không chứa answer, giải mã span. Metric cũng đổi từ accuracy/F1 theo lớp sang EM và F1 theo token của answer.

## 2. Extractive QA và RAG

Extractive QA: Question + Context -> Answer span. Context được cho sẵn, model chỉ chỉ ra vị trí đáp án trong context đó, không sinh từ mới.

RAG: Question -> Retriever -> Relevant documents -> Context -> Generator / QA model -> Answer. Hệ thống phải tự tìm context trong kho tài liệu lớn, sau đó sinh hoặc trích câu trả lời.

Khác biệt chính:

- Extractive QA giả định context đúng đã có, RAG phải giải bài toán tìm context (retrieval) trước.
- Extractive QA chỉ trả về span có sẵn trong văn bản, RAG có thể dùng generator để tổng hợp từ nhiều tài liệu và diễn đạt lại.
- Lỗi của extractive QA nằm ở ranh giới span, lỗi của RAG có thêm lỗi retrieval (lấy sai tài liệu) và hallucination của generator.

## 3. No-answer (SQuAD 2.0)

Extractive QA trên SQuAD 1.1 luôn trả về một span, kể cả khi context không chứa câu trả lời. Trong document_qa.py, câu "Who is the president of France?" vẫn nhận được một span sai với score thấp. SQuAD 2.0 xử lý bằng cách cho phép model trỏ vào [CLS] (start = end = 0) khi không có answer, và so sánh score của span tốt nhất với score null cộng một ngưỡng. Trong lab này chỉ dùng ngưỡng score đơn giản để cảnh báo độ tin cậy thấp.

## 4. Individual Learning Check

1. Extractive QA khác classification như thế nào?
Classification gán một nhãn từ tập nhãn cố định cho cả input, dùng vector [CLS]. Extractive QA nhận question + context và trả về một đoạn (start, end) trong context, output space là các vị trí và thay đổi theo từng context.

2. Tại sao QA cần start và end position?
Answer là một span liên tục, được xác định hoàn toàn bởi vị trí đầu và vị trí cuối. Dự đoán hai phân phối trên T vị trí chỉ cần 2T output thay vì T(T+1)/2 span.

3. Character offset khác token index như thế nào?
Character offset là vị trí ký tự trong chuỗi gốc (Paris bắt đầu ở ký tự 25), token index là vị trí trong chuỗi input_ids sau tokenization, gồm cả [CLS], question, [SEP] và subword (Paris là token 14).

4. Tại sao cần offset mapping?
SQuAD cho answer theo ký tự còn model học trên token. Offset mapping cho biết mỗi token ứng với đoạn ký tự nào, dùng để đổi nhãn ký tự sang token khi train và đổi span token dự đoán về đoạn văn bản gốc khi suy luận.

5. stride dùng để làm gì?
stride là số token overlap giữa hai window liên tiếp khi cắt context dài, để answer nằm gần ranh giới window vẫn xuất hiện trọn vẹn trong ít nhất một window.

6. Vì sao context dài có thể phải chia thành nhiều window?
Transformer có max sequence length (512 với BERT, thường dùng 384) và chi phí attention tăng theo bình phương độ dài. Context dài hơn phải chia window, nếu chỉ truncate thì answer ở phần bị cắt sẽ mất.

7. Exact Match đo điều gì?
Tỉ lệ prediction trùng hoàn toàn với ít nhất một gold answer sau normalization (lowercase, bỏ dấu câu, bỏ a/an/the, chuẩn hóa khoảng trắng).

8. F1 của QA khác gì với accuracy?
F1 của QA tính cho từng câu hỏi dựa trên overlap token giữa prediction và gold (precision, recall), cho điểm một phần, rồi lấy trung bình. Accuracy chỉ đúng/sai theo nhãn, gần với EM hơn.

9. Vì sao prediction có thể có F1 > 0 nhưng EM = 0?
Khi prediction chồng lấn một phần với gold: thiếu hoặc thừa vài từ ("New York" so với "New York City" có F1 = 0.8, EM = 0).

10. Wrong-start và wrong-end khác nhau như thế nào?
Wrong-start: vị trí bắt đầu sai, vị trí kết thúc đúng (thiếu hoặc thừa phần đầu, ví dụ "York City"). Wrong-end: bắt đầu đúng, kết thúc sai (thiếu hoặc thừa phần cuối, ví dụ "New York"). Hai lỗi đến từ hai classifier khác nhau.

11. Extractive QA khác RAG ở đâu?
Extractive QA cần context cho sẵn và chỉ trích span. RAG tự tìm context bằng retriever trong kho tài liệu lớn rồi dùng generator (hoặc QA model) để trả lời.

12. Khi nào extractive QA không thể trả lời câu hỏi?
Khi context không chứa câu trả lời, khi câu trả lời không phải một đoạn liên tục của văn bản (yes/no, đếm, tổng hợp nhiều chỗ, cần suy luận hoặc tính toán), khi answer nằm ở phần bị truncate, hoặc khi chưa biết context nào chứa câu trả lời (cần retrieval).
