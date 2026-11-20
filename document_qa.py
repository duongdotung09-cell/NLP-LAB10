"""
LAB 10 - Mini Document QA
User question -> context/document -> fine-tuned QA model -> answer span

Usage:
    python document_qa.py                      # chạy các câu hỏi demo trên tài liệu mẫu
    python document_qa.py -i                   # hỏi đáp tương tác trên tài liệu mẫu
    python document_qa.py -f my_doc.txt -i     # hỏi đáp trên tài liệu riêng
    python document_qa.py -q "When was SQuAD released?"
"""
import argparse
import os

import numpy as np
import torch

import qa_model as qm

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "outputs", "len384", "model")

DOCUMENT = """The Transformer is a deep learning architecture introduced in the paper "Attention Is All You Need", published in 2017 by Ashish Vaswani and seven co-authors at Google. It replaced recurrence with self-attention, which lets every token look at every other token in a sequence. Because the computation over tokens can run in parallel, Transformers train much faster on GPUs than recurrent neural networks.

BERT, which stands for Bidirectional Encoder Representations from Transformers, was released by Google in October 2018. BERT is pretrained with two objectives: masked language modeling, in which 15 percent of the input tokens are hidden and must be predicted, and next sentence prediction. The base version of BERT has 12 layers and about 110 million parameters, while the large version has 24 layers and about 340 million parameters.

DistilBERT is a smaller model created by Hugging Face in 2019 using knowledge distillation. It keeps 6 of the 12 layers of BERT base, has about 66 million parameters, runs 60 percent faster and retains about 97 percent of the language understanding performance of BERT. Unlike BERT, DistilBERT does not use token type embeddings.

The Stanford Question Answering Dataset, known as SQuAD, was released in 2016 by Pranav Rajpurkar and colleagues at Stanford University. SQuAD 1.1 contains more than 100,000 questions written by crowdworkers on 536 Wikipedia articles. Every answer is a span of text copied from the corresponding paragraph, so the task is called extractive question answering. SQuAD 2.0, released in 2018, added over 50,000 unanswerable questions that look similar to answerable ones.

Systems on SQuAD are evaluated with two metrics. Exact Match measures the percentage of predictions that match one of the gold answers exactly after normalization. The F1 score measures the average overlap between the tokens of the prediction and the tokens of the gold answer. On SQuAD 1.1, human performance was reported as 82.3 Exact Match and 91.2 F1.

In this course, Lab 10 covers extractive question answering and must be submitted before 23:59 on 25 November 2026. Lab 11 moves to semantic retrieval, where the system must first find relevant documents in a large collection, and Lab 12 combines retrieval with generation to build a retrieval-augmented generation system, usually called RAG."""

DEMO_QUESTIONS = [
    "Who introduced the Transformer architecture?",
    "In which year was BERT released?",
    "How many parameters does BERT large have?",
    "What percentage of input tokens are masked in BERT pretraining?",
    "How many layers does DistilBERT keep?",
    "How much faster is DistilBERT than BERT?",
    "How many Wikipedia articles are used in SQuAD 1.1?",
    "What did SQuAD 2.0 add?",
    "What does Exact Match measure?",
    "What was the human F1 score on SQuAD 1.1?",
    "When is the deadline of Lab 10?",
    "What does Lab 11 focus on?",
    "Which company created DistilBERT?",
    "Who is the president of France?",  # không có trong tài liệu
]


class DocumentQA:
    def __init__(self, model_path=DEFAULT_MODEL, max_length=384, stride=128,
                 min_score=0.0):
        if not os.path.isdir(model_path):
            raise SystemExit(
                f"Không tìm thấy model tại {model_path}. "
                "Hãy chạy fine-tuning trước: python qa_model.py 384 128 1")
        self.tok, self.model = qm.load_model(model_path)
        self.model.eval()
        self.max_length, self.stride = max_length, stride
        self.min_score = min_score

    def ask(self, question, document):
        pred = qm.answer_question(question, document, self.tok, self.model,
                                  self.max_length, self.stride)
        pred["confident"] = pred["score"] >= self.min_score
        return pred


def show(pred, question, document):
    print(f"Q: {question}")
    if not pred["text"]:
        print("A: (không tìm được span hợp lệ)\n")
        return
    s, e = pred["start_char"], pred["end_char"]
    # sentence chứa answer để người dùng kiểm tra
    left = max(document.rfind(". ", 0, s) + 2, document.rfind("\n", 0, s) + 1)
    ends = [p for p in (document.find(". ", e), document.find("\n", e)) if p != -1]
    right = min(ends) + 1 if ends else len(document)
    evidence = document[left:s] + "[[" + document[s:e] + "]]" + document[e:right]
    flag = "" if pred["confident"] else "  (độ tin cậy thấp, có thể tài liệu không chứa câu trả lời)"
    print(f"A: {pred['text']}   score={pred['score']:.2f} chars=[{s},{e}){flag}")
    print(f"   evidence: {evidence.strip()}\n")


def main():
    ap = argparse.ArgumentParser(description="Mini Document QA (extractive)")
    ap.add_argument("-f", "--file", help="file .txt chứa tài liệu")
    ap.add_argument("-q", "--question", help="một câu hỏi")
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL)
    ap.add_argument("--min-score", type=float, default=5.0,
                    help="ngưỡng start_logit + end_logit để coi là tin cậy")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    document = DOCUMENT
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            document = f.read()
    qa = DocumentQA(args.model, min_score=args.min_score)
    n_tok = len(qa.tok(document)["input_ids"])
    print(f"Document: {len(document)} ký tự, {n_tok} token "
          f"-> sliding window max_length={qa.max_length}, stride={qa.stride}\n")

    if args.question:
        show(qa.ask(args.question, document), args.question, document)
    elif args.interactive:
        while True:
            try:
                q = input("Question (Enter để thoát): ").strip()
            except EOFError:
                break
            if not q:
                break
            show(qa.ask(q, document), q, document)
    else:
        for q in DEMO_QUESTIONS:
            show(qa.ask(q, document), q, document)


if __name__ == "__main__":
    main()
