"""
LAB 10 - Extractive Question Answering
Core module: preprocessing (offset mapping / span alignment), model training,
post-processing (logits -> answer span), EM/F1 metrics and keyword baseline.
"""
import collections
import math
import random
import re
import string
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForQuestionAnswering, AutoTokenizer,
                          get_linear_schedule_with_warmup)

MODEL_NAME = "distilbert-base-uncased"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# 1. Span alignment: character span -> token span
# ---------------------------------------------------------------------------
def char_span_to_token_span(offsets, sequence_ids, answer_start, answer_end):
    """
    offsets: list of (start_char, end_char) for every token in one feature.
    sequence_ids: 0 = question, 1 = context, None = special token.
    answer_start: index of first char of the answer in context.
    answer_end: index right after the last char (exclusive).
    Return (start_token, end_token) or None if the answer is not fully inside
    the context part of this window.
    """
    ctx = [i for i, s in enumerate(sequence_ids) if s == 1]
    if not ctx or answer_end <= answer_start:  # no context / empty answer
        return None
    win_start, win_end = offsets[ctx[0]][0], offsets[ctx[-1]][1]
    # answer truncated / outside this window
    if answer_start < win_start or answer_end > win_end:
        return None

    start_token, end_token = None, None
    for token_idx in ctx:
        start, end = offsets[token_idx]
        # start <= answer_start < end : answer starts inside this token
        # (also covers answer starting in the middle of a token)
        if start_token is None and start <= answer_start < end:
            start_token = token_idx
        # start < answer_end <= end : answer ends inside this token
        # (also covers answer ending in the middle of a token)
        if start < answer_end <= end:
            end_token = token_idx
    # answer_start/answer_end fall on whitespace between tokens
    if start_token is None:
        start_token = next(i for i in ctx if offsets[i][0] >= answer_start)
    if end_token is None:
        end_token = [i for i in ctx if offsets[i][1] <= answer_end][-1]
    return start_token, end_token


def _context_bounds(sequence_ids):
    ctx_start = sequence_ids.index(1)
    ctx_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)
    return ctx_start, ctx_end


# ---------------------------------------------------------------------------
# 2. Feature preparation
# ---------------------------------------------------------------------------
def prepare_features(examples, tokenizer, max_length=384, stride=128):
    """
    Tokenize (question, context) with sliding windows.
    Returns features with start_positions/end_positions (CLS=0 if the answer is
    not in the window) and the information needed for post-processing.
    """
    questions = [q.strip() for q in examples["question"]]
    enc = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,  # dynamic padding in the collate function
    )
    sample_map = enc.pop("overflow_to_sample_mapping")
    enc["start_positions"], enc["end_positions"] = [], []
    enc["example_id"], enc["answer_in_window"] = [], []

    for i, offsets in enumerate(enc["offset_mapping"]):
        seq_ids = enc.sequence_ids(i)
        ex_idx = sample_map[i]
        ans = examples["answers"][ex_idx]
        a_start = ans["answer_start"][0]
        a_end = a_start + len(ans["text"][0])
        span = char_span_to_token_span(offsets, seq_ids, a_start, a_end)
        if span is None:
            enc["start_positions"].append(0)
            enc["end_positions"].append(0)
            enc["answer_in_window"].append(False)
        else:
            enc["start_positions"].append(span[0])
            enc["end_positions"].append(span[1])
            enc["answer_in_window"].append(True)
        enc["example_id"].append(examples["id"][ex_idx])
        # keep offsets only for context tokens (used in post-processing)
        enc["offset_mapping"][i] = [
            o if seq_ids[k] == 1 else None for k, o in enumerate(offsets)
        ]
    return enc


def build_features(dataset, tokenizer, max_length=384, stride=128):
    return dataset.map(
        lambda b: prepare_features(b, tokenizer, max_length, stride),
        batched=True,
        remove_columns=dataset.column_names,
    )


PAD_ID = 0  # [PAD] id of BERT/DistilBERT uncased vocab


def _pad(seqs, value):
    L = max(len(s) for s in seqs)
    return torch.tensor([list(s) + [value] * (L - len(s)) for s in seqs])


def _collate(batch, with_labels=True):
    """Dynamic padding: pad every batch to its longest feature."""
    out = {"input_ids": _pad([b["input_ids"] for b in batch], PAD_ID),
           "attention_mask": _pad([b["attention_mask"] for b in batch], 0)}
    if with_labels:
        out["start_positions"] = torch.tensor([b["start_positions"] for b in batch])
        out["end_positions"] = torch.tensor([b["end_positions"] for b in batch])
    return out


# ---------------------------------------------------------------------------
# 3. Training
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_loss(model, features, batch_size=32):
    model.eval()
    loader = DataLoader(features, batch_size=batch_size, collate_fn=_collate)
    total, n = 0.0, 0
    for batch in loader:
        out = model(**batch)
        total += out.loss.item() * batch["input_ids"].size(0)
        n += batch["input_ids"].size(0)
    model.train()
    return total / n


def train(model, train_feats, val_feats, epochs=2, lr=5e-5, batch_size=16,
          warmup_ratio=0.1, weight_decay=0.01, eval_every=None, seed=42,
          log=print):
    """
    QA loss (HF): L = (CE(start_logits, i) + CE(end_logits, j)) / 2
    Returns history dict with training/validation loss.
    """
    set_seed(seed)
    loader = DataLoader(train_feats, batch_size=batch_size, shuffle=True,
                        collate_fn=_collate,
                        generator=torch.Generator().manual_seed(seed))
    total_steps = epochs * len(loader)
    eval_every = eval_every or max(1, len(loader) // 4)
    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optim = torch.optim.AdamW(params, lr=lr)
    sched = get_linear_schedule_with_warmup(
        optim, int(warmup_ratio * total_steps), total_steps)

    hist = {"step": [], "epoch": [], "train_loss": [], "val_loss": []}
    model.train()
    step, running, n_run = 0, 0.0, 0
    t0 = time.time()
    for ep in range(epochs):
        for batch in loader:
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
            step += 1
            running += out.loss.item()
            n_run += 1
            if step % eval_every == 0 or step == total_steps:
                val_loss = compute_loss(model, val_feats)
                hist["step"].append(step)
                hist["epoch"].append(step / len(loader))
                hist["train_loss"].append(running / n_run)
                hist["val_loss"].append(val_loss)
                log(f"step {step}/{total_steps} epoch {step/len(loader):.2f} "
                    f"train_loss {running/n_run:.4f} val_loss {val_loss:.4f} "
                    f"({time.time()-t0:.0f}s)")
                running, n_run = 0.0, 0
    hist["train_time_s"] = time.time() - t0
    return hist


# ---------------------------------------------------------------------------
# 4. Inference + post-processing
# ---------------------------------------------------------------------------
@torch.no_grad()
def get_logits(model, features, batch_size=64):
    model.eval()
    loader = DataLoader(features, batch_size=batch_size,
                        collate_fn=lambda b: _collate(b, with_labels=False))
    starts, ends = [], []
    for batch in loader:
        out = model(**batch)
        # pad logits back to a common length so they can be concatenated
        L = batch["input_ids"].size(1)
        s = np.full((out.start_logits.size(0), 512), -1e4, dtype=np.float32)
        e = np.full_like(s, -1e4)
        s[:, :L] = out.start_logits.numpy()
        e[:, :L] = out.end_logits.numpy()
        starts.append(s)
        ends.append(e)
    return np.concatenate(starts), np.concatenate(ends)


def postprocess(examples, features, start_logits, end_logits, n_best=20,
                max_answer_len=30):
    """
    For every example: look at all its windows, consider top n_best start and
    end indices, keep valid spans (inside context, start <= end, length limit),
    choose the span with the highest start_logit + end_logit.
    Returns dict id -> {text, start_char, end_char, score}.
    """
    feat_per_ex = collections.defaultdict(list)
    for i, ex_id in enumerate(features["example_id"]):
        feat_per_ex[ex_id].append(i)
    offsets_all = features["offset_mapping"]

    preds = {}
    for ex in examples:
        best = {"text": "", "start_char": -1, "end_char": -1, "score": -1e9}
        for fi in feat_per_ex[ex["id"]]:
            offsets = offsets_all[fi]
            s_log = start_logits[fi][:len(offsets)]
            e_log = end_logits[fi][:len(offsets)]
            s_idx = np.argsort(s_log)[-1:-n_best - 1:-1]
            e_idx = np.argsort(e_log)[-1:-n_best - 1:-1]
            for s in s_idx:
                for e in e_idx:
                    if offsets[s] is None or offsets[e] is None:
                        continue
                    if e < s or e - s + 1 > max_answer_len:
                        continue
                    score = s_log[s] + e_log[e]
                    if score > best["score"]:
                        sc, ec = offsets[s][0], offsets[e][1]
                        best = {"text": ex["context"][sc:ec],
                                "start_char": sc, "end_char": ec,
                                "score": float(score)}
        preds[ex["id"]] = best
    return preds


# ---------------------------------------------------------------------------
# 5. Metrics (SQuAD-style normalization)
# ---------------------------------------------------------------------------
def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred, gold):
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred, gold):
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    common = collections.Counter(p_toks) & collections.Counter(g_toks)
    overlap = sum(common.values())
    if len(p_toks) == 0 or len(g_toks) == 0:
        return float(p_toks == g_toks)
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_toks)
    recall = overlap / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def score_example(pred, gold_answers):
    """SQuAD: take the max over all gold answers."""
    em = max(exact_match(pred, g) for g in gold_answers)
    f1 = max(f1_score(pred, g) for g in gold_answers)
    return em, f1


def evaluate(examples, pred_texts):
    ems, f1s = [], []
    for ex in examples:
        em, f1 = score_example(pred_texts[ex["id"]], ex["answers"]["text"])
        ems.append(em)
        f1s.append(f1)
    return {"EM": 100 * np.mean(ems), "F1": 100 * np.mean(f1s),
            "em_list": ems, "f1_list": f1s}


# ---------------------------------------------------------------------------
# 6. Baseline: keyword overlap -> candidate sentence -> candidate span
# ---------------------------------------------------------------------------
STOPWORDS = set("""a an the of in on at to for from by with and or is are was
were be been being what which who whom whose when where why how did do does
that this these those it its as into than then there their they he she his her
him we you your i me my our not no can could would should will shall may might
has have had about after before during over under between up down out""".split())

_WORD_RE = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]")


def _sentences(context):
    spans, start = [], 0
    for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", context):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(context)))
    return spans


def _content(word):
    w = word.lower()
    return w if (w not in STOPWORDS and re.match(r"\w", w)) else None


def baseline_predict(question, context, max_span=5):
    """
    1) choose the sentence with the largest keyword overlap with the question;
    2) in that sentence, candidate spans = maximal runs of tokens that do not
       appear in the question (stop at punctuation);
    3) choose the candidate closest to the matched question words, truncated
       to max_span tokens.
    """
    q_words = {_content(w) for w in _WORD_RE.findall(question)} - {None}
    best_sent, best_score = None, -1
    for s, e in _sentences(context):
        words = {_content(w) for w in _WORD_RE.findall(context[s:e])} - {None}
        score = len(words & q_words)
        if score > best_score:
            best_sent, best_score = (s, e), score
    s0, e0 = best_sent
    toks = [(m.start() + s0, m.end() + s0, m.group())
            for m in _WORD_RE.finditer(context[s0:e0])]
    matched = [i for i, t in enumerate(toks) if _content(t[2]) in q_words]

    candidates, cur = [], []
    for i, (_, _, w) in enumerate(toks):
        is_punct = not re.match(r"\w", w)
        in_q = w.lower() in q_words or w.lower() in STOPWORDS
        if is_punct or in_q:
            if cur:
                candidates.append(cur)
            cur = []
        else:
            cur.append(i)
    if cur:
        candidates.append(cur)
    if not candidates:
        return ""

    def dist(c):
        if not matched:
            return 0
        return min(abs(i - m) for i in (c[0], c[-1]) for m in matched)

    best = min(candidates, key=lambda c: (dist(c), -len(c)))
    best = best[:max_span]
    return context[toks[best[0]][0]:toks[best[-1]][1]]


# ---------------------------------------------------------------------------
# 7. Data splits + full experiment (used by qa_finetuning.ipynb)
# ---------------------------------------------------------------------------
N_TRAIN, N_VAL, N_TEST = 3000, 300, 1000
SEED = 42


def load_splits(n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, seed=SEED):
    """
    train : n_train random examples of SQuAD v1.1 train
    val   : n_val random examples of SQuAD v1.1 validation (validation loss)
    test  : n_test other random examples of SQuAD v1.1 validation (EM/F1)
    """
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad")
    train = ds["train"].shuffle(seed=seed).select(range(n_train))
    dev = ds["validation"].shuffle(seed=seed)
    test = dev.select(range(n_test))
    val = dev.select(range(n_test, n_test + n_val))
    return train, val, test


def run_experiment(out_dir, max_length=384, stride=128, epochs=2, lr=5e-5,
                   batch_size=16, seed=SEED, save_model=False):
    """Fine-tune, evaluate on test, save history/predictions/metrics."""
    import json
    import os
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)
    train_ds, val_ds, test_ds = load_splits()
    tok, model = load_model(MODEL_NAME)
    tr_f = build_features(train_ds, tok, max_length, stride)
    va_f = build_features(val_ds, tok, max_length, stride)
    te_f = build_features(test_ds, tok, max_length, stride)
    print(f"max_length={max_length} stride={stride} train_feats={len(tr_f)} "
          f"val_feats={len(va_f)} test_feats={len(te_f)}", flush=True)
    hist = train(model, tr_f, va_f, epochs=epochs, lr=lr,
                 batch_size=batch_size, seed=seed,
                 log=lambda m: print(m, flush=True))
    t0 = time.time()
    s, e = get_logits(model, te_f)
    preds = postprocess(test_ds, te_f, s, e)
    infer_time = time.time() - t0
    res = evaluate(test_ds, {k: v["text"] for k, v in preds.items()})
    metrics = {"max_length": max_length, "stride": stride, "epochs": epochs,
               "lr": lr, "batch_size": batch_size, "seed": seed,
               "n_train": len(train_ds), "n_train_features": len(tr_f),
               "n_test": len(test_ds), "EM": res["EM"], "F1": res["F1"],
               "train_time_s": hist["train_time_s"],
               "test_infer_time_s": infer_time}
    print(metrics, flush=True)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(hist, f, indent=1)
    with open(os.path.join(out_dir, "test_predictions.json"), "w",
              encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    if save_model:
        model.save_pretrained(os.path.join(out_dir, "model"))
        tok.save_pretrained(os.path.join(out_dir, "model"))
    return metrics


# ---------------------------------------------------------------------------
# 8. Convenience for loading model/tokenizer
# ---------------------------------------------------------------------------
def load_model(name_or_path=MODEL_NAME):
    tok = AutoTokenizer.from_pretrained(name_or_path)
    model = AutoModelForQuestionAnswering.from_pretrained(name_or_path)
    return tok, model


def answer_question(question, context, tokenizer, model, max_length=384,
                    stride=128, n_best=20, max_answer_len=30):
    """Run the full pipeline (tokenize -> logits -> best span) for one pair."""
    from datasets import Dataset
    ex = Dataset.from_dict({"id": ["q0"], "question": [question],
                            "context": [context],
                            "answers": [{"text": [""], "answer_start": [0]}]})
    feats = build_features(ex, tokenizer, max_length, stride)
    s, e = get_logits(model, feats)
    return postprocess(ex, feats, s, e, n_best, max_answer_len)["q0"]


if __name__ == "__main__":
    # python qa_model.py 384 128 1   -> max_length, stride, save_model
    import sys
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 384
    S = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    save = len(sys.argv) > 3 and sys.argv[3] == "1"
    torch.set_num_threads(torch.get_num_threads())
    run_experiment(f"outputs/len{L}", max_length=L, stride=S, save_model=save)
