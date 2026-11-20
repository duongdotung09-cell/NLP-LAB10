# Error Analysis - LAB 10

Model: DistilBERT fine-tune trên 3.000 câu SQuAD train (max_length 384, stride 128), test trên 1.000 câu SQuAD validation: EM 46.7, F1 57.5. Dữ liệu đầy đủ trong outputs/errors_fine_tuned.csv (sinh từ qa_finetuning.ipynb, mục 9).

## 1. Cách phân loại

Với mỗi câu EM = 0, chọn gold answer có F1 cao nhất với prediction, bỏ dấu câu và khoảng trắng ở hai đầu cả gold lẫn prediction rồi so vị trí ký tự:

- wrong start: end đúng, start sai.
- wrong end: start đúng, end sai.
- wrong boundary: cả hai đầu sai nhưng prediction vẫn chồng lấn gold.
- wrong span: prediction không chồng lấn gold (chọn sai chỗ trong context).

| Loại | Số câu | Tỉ lệ trên 533 câu sai |
|---|---|---|
| wrong start | 83 | 15.6% |
| wrong end | 79 | 14.8% |
| wrong boundary (sai cả hai đầu, có overlap) | 30 | 5.6% |
| wrong span (không overlap) | 341 | 64.0% |
| Tổng EM = 0 | 533 | 100% |

## 2. Bảng lỗi

| # | Question | Gold | Prediction | Error type | Explanation |
|---|---|---|---|---|---|
| 1 | At what address did Goldenson secure a new headquarters for ABC? | 1330 Avenue of the Americas in Manhattan | 44 story building located at 1330 Avenue of the Americas in Manhattan | wrong start | Start lùi về cụm "44 story building" vì nó đứng ngay sau "headquarters" (từ trong question). Thừa phần đầu, end đúng. |
| 2 | During what period did the Tower Theatre re-open? | late 1970s | 1970s | wrong start | Model chỉ lấy thực thể thời gian chuẩn "1970s", bỏ bổ ngữ "late". Partial answer. |
| 3 | How many Huguenots were killed in Toulouse? | Nearly 3,000 | 3,000 | wrong start | Câu hỏi "how many" làm model ưu tiên bắt đầu ở con số, bỏ trạng từ "Nearly". Partial answer. |
| 4 | What is the name of the port city that was created? | Mombasa | Swahili built Mombasa | wrong start | Start lùi về chủ ngữ "Swahili", một danh từ riêng khác đứng ngay trước đáp án; end đúng nên thừa phần đầu. Hai tên riêng đứng gần nhau làm start head khó chọn ranh giới. |
| 5 | What are the only states where ABC doesn't have a licensed affiliate? | New Jersey, Rhode Island and Delaware | Delaware | wrong start | Đáp án là danh sách, model chỉ lấy phần tử cuối đứng sát "are the only U.S. states". Partial answer, start head chưa học được span liệt kê. |
| 6 | When did Zhu Shijie die? | 1314 | 1249–1314 | wrong start (tokenization) | "(1249–1314)" không có khoảng trắng nên sau normalization thành một token "1249–1314", F1 = 0 dù end đúng. Lỗi do tokenization/normalization và model không hiểu "die" ứng với năm thứ hai. |
| 7 | What studio does ABC own at 1500 Broadway in NYC? | Times Square Studios | Times Square | wrong end | End dừng sớm ở "Square". "Times Square" xuất hiện hai lần trong câu (tên studio và địa danh), model coi là địa danh. Multiple plausible spans. |
| 8 | How many fraternities are apart of the university? | fifteen | fifteen fraternities and seven | wrong end | End kéo dài sang "and seven" vì cấu trúc liệt kê "fifteen fraternities and seven sororities". Wrong boundary. |
| 9 | What is 'grey literature'? | non-peer-reviewed sources | non-peer-reviewed sources (the "grey literature"), provided that they are of sufficient quality | wrong end | End nhảy qua ngoặc đơn tới hết mệnh đề. Câu hỏi định nghĩa không có dấu hiệu độ dài, end head chọn ranh giới câu. |
| 10 | Which university has its origins in a school dealing with medicine and surgery? | Newcastle University | Newcastle University has its origins in the School of Medicine and Surgery | wrong end | Model trả về gần cả câu vì phần sau trùng gần hết với question (lexical overlap). Start đúng, end bị kéo về cuối cụm trùng. |
| 11 | What types of medications do specialty pharmacies stock? | novel medications | novel | wrong end | End dừng ở tính từ, bỏ danh từ "medications" vì từ này đã có trong question. Partial answer. |
| 12 | Who ends up getting more of the teacher's resources in this scenario? | motivated students | motivated students, ignoring attention-seeking and disruptive students. The result of this is that motivated students, facing demanding university entrance examinations | wrong end | "motivated students" xuất hiện hai lần, end chọn lần xuất hiện thứ hai, span vượt qua dấu chấm câu. Multiple plausible spans + span quá dài (gần max_answer_len). |
| 13 | One of the sites, Merceds-Benz Superdome, is located where? | New Orleans | New Orleans' Mercedes-Benz Superdome, Miami's Sun Life Stadium | wrong end | End rơi vào danh sách các địa điểm phía sau. Question có lỗi chính tả "Merceds" nên khớp kém. |
| 14 | What are other alternative names for French and Indian War? | the Fourth Intercolonial War and the Great War for the Empire | the Fourth Intercolonial War | wrong end | Đáp án gồm hai tên nối bằng "and", model chỉ lấy tên đầu. Partial answer. |
| 15 | Name one country that banned boating, driving and flying on Sundays. | Germany / Italy / Norway | The UK, Germany, Italy, Switzerland and Norway | wrong boundary | Question chỉ cần một nước nhưng model lấy cả danh sách. Question ambiguity: mọi nước trong danh sách đều đúng. |
| 16 | What culture's arrival in Scotland is know as the "Davidian Revolution"? | Norman | King David I of Scotland, ... introducing Normans and Norman culture | wrong boundary | Câu dài nhiều mệnh đề, model lấy gần như cả câu. Long context / span quá dài. |
| 17 | What series years are available on Blu-ray? | from 2009 onwards | 2009 | wrong boundary | Model chỉ lấy con số, bỏ "from ... onwards". Partial answer. |
| 18 | Why was this short termed organization created? | to coordinate the response to the embargo | William E. Simon | wrong span | Câu hỏi "why" nhưng model trả tên người nằm cạnh "short-term organization". Model chưa phân biệt loại câu hỏi, chọn thực thể gần từ khóa. |
| 19 | What does LGM stands for? | Last Glacial Maximum | deglaciation | wrong span | Đáp án đứng ngay trước "(LGM)", model chọn từ ngay sau. Ít ví dụ dạng viết tắt trong 3.000 câu train. |
| 20 | How many teams up to Super Bowl 50 have been to the championship game eight times? | four | the Denver Broncos | wrong span | Câu "the Denver Broncos became one of four teams to have made eight appearances": model trả về chủ ngữ thay vì số. Distractor, sai loại thực thể. |
| 21 | Approximately how many items are in the costume collection of the V&A? | over 14,000 | 1,442 | wrong span | Đoạn văn có hai con số về costume, model chọn số đi cùng "costumes and items" (trùng từ "items" của question). Distractor + long context. |
| 22 | How many people were on the test flight of the AS-206? | unmanned | 278 | wrong span | Câu hỏi "how many" nhưng đáp án là chữ "unmanned". Model chọn số trong "AS-278". Tokenization tách "AS-278" thành nhiều subword nên "278" thành một candidate. |
| 23 | For how long did the plague stick around? | until the 19th century | 30–60% | wrong span | Context dài với nhiều mốc và con số, câu chứa đáp án nằm cuối đoạn. Long context + distractor. |
| 24 | Who did Tesla think would run the world of the future? | women | Queen Bees | wrong span | "Queen Bees" là cách nói ẩn dụ khớp với "run the world of the future", đáp án đúng nằm ở câu sau. Multiple plausible spans / cần suy luận qua câu. |
| 25 | The point at which different radiometric isotopes stop diffusing ... is called what? | closure temperature | different radiometric isotopes stop diffusing into and out of the crystal lattice | wrong span | Model chép lại phần context trùng với question thay vì tên thuật ngữ đứng trước. Phụ thuộc lexical overlap. |

## 3. Ví dụ EM = 0 nhưng F1 > 0 (Experiment 3)

| Question | Gold | Prediction | P | R | F1 | Vì sao EM = 0 |
|---|---|---|---|---|---|---|
| What studio does ABC own at 1500 Broadway in NYC? | Times Square Studios | Times Square | 2/2 | 2/3 | 0.80 | Thiếu 1 token cuối |
| How many Huguenots were killed in Toulouse? | Nearly 3,000 | 3,000 | 1/1 | 1/2 | 0.67 | Thiếu trạng từ "Nearly" |
| What kind of light is important for chloroplasts to divide? | white light / bright white light | bright white | 2/2 | 2/3 | 0.80 | Thiếu "light" |
| How many fraternities are apart of the university? | fifteen | fifteen fraternities and seven | 1/4 | 1/1 | 0.40 | Thừa 3 token |
| What monarchy did western troops protect? | Saudi | Saudi monarchy | 1/2 | 1/1 | 0.67 | Thừa "monarchy" (gold chỉ là tính từ) |
| What is 'grey literature'? | non-peer-reviewed sources | non-peer-reviewed sources (the "grey literature"), provided ... quality | 2/11 | 2/2 | 0.31 | Span quá dài |

Tính mẫu dòng đầu: overlap = {times, square} = 2, P = 2/2 = 1.0, R = 2/3 = 0.667, F1 = 2 x 1.0 x 0.667 / 1.667 = 0.80.

EM chỉ cho điểm khi trùng hoàn toàn, F1 cho điểm một phần theo overlap. Các lỗi ranh giới (thiếu/thừa vài token) có EM = 0 nhưng F1 từ 0.3 đến 0.8. Đây là lý do F1 (57.5) cao hơn EM (46.7) khoảng 11 điểm.

## 4. Wrong start vs wrong end

| | Số câu |
|---|---|
| #wrong start (start sai, tính cả câu sai cả hai đầu) | 454 |
| #wrong end (end sai, tính cả câu sai cả hai đầu) | 450 |
| Chỉ sai start | 83 |
| Chỉ sai end | 79 |
| Trong các câu có overlap: prediction dài hơn gold / ngắn hơn hoặc lệch | 115 / 77 |

Nhận xét:

- 341/533 lỗi là wrong span, cả start và end cùng sai. Vì vậy #wrong start và #wrong end gần bằng nhau (454 và 450): khi chọn sai vị trí thì cả hai head cùng sai. Lỗi chính của model là chọn sai chỗ chứ không phải lệch ranh giới.
- Xét riêng lỗi ranh giới, chỉ sai start (83) và chỉ sai end (79) gần như ngang nhau. Prediction thường dài hơn gold (115 so với 77).

Giả thuyết nguyên nhân:

- Wrong start thường do bổ ngữ đứng trước đáp án ("Nearly", "late", "the political weakness of") và do gold của SQuAD không thống nhất có lấy mạo từ, giới từ hay bổ ngữ hay không. Start head có xu hướng bắt đầu ở thực thể "chuẩn" (con số, danh từ riêng).
- Wrong end thường do danh sách, mệnh đề phụ, ngoặc đơn sau đáp án. End phải quyết định dừng ở đâu nên mơ hồ hơn. Ngoài ra start và end được dự đoán độc lập, chỉ ràng buộc khi giải mã (i ≤ j, độ dài ≤ 30), nên end dễ nhảy tới một vị trí có score cao ở xa (ví dụ 12).
- Model chỉ train 3.000 câu nên còn dựa nhiều vào lexical overlap: span hay bắt đầu hoặc kết thúc ở chỗ trùng từ với question (ví dụ 1, 10, 25).
- Annotation: nhiều câu có nhiều gold khác nhau về ranh giới ("Mughal state" và "the Mughal state"), chứng tỏ bản thân ranh giới đã mơ hồ, nên lỗi ranh giới một phần là không thể tránh.

## 5. Tổng hợp theo loại lỗi

- distractor / multiple plausible spans: lỗi phổ biến nhất trong nhóm wrong span, đặc biệt câu hỏi "how many", "when" có nhiều con số, năm trong cùng đoạn.
- partial answer: đa số lỗi wrong start / wrong end, F1 vẫn > 0.
- question ambiguity: câu hỏi cho phép nhiều đáp án (ví dụ 15).
- long context: context dài nhiều mệnh đề làm span quá dài hoặc chọn sai câu (ví dụ 16, 23).
- tokenization: chuỗi dính liền như "1249–1314", "AS-278" làm ranh giới token không khớp với gold (ví dụ 6, 22).
- truncation: không gặp trong mẫu này. Với max_length 384 chỉ khoảng 2% test example dài hơn một window và stride 128 đảm bảo answer nằm trọn trong một window.
