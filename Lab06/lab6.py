"""Lab 6: NLP Fundamentals - Tokenization, Embeddings and Text Chunking (single-file version).

Everything from the manual is in this one file: shared Retriever (TF-IDF or real sentence
embeddings), token comparison (Task 1), FAQ chatbot, chunking with source offsets (Task 2),
retrieval evaluation with Hit@3 / MRR@3 (Task 3), document chatbot, and the five scenarios.
All documents, rules and queries are SYNTHETIC teaching data, not real policies.

Setup : python -m pip install numpy scikit-learn sentence-transformers
Run   : python lab6.py all                      (all tasks; uses embeddings if available)
        python lab6.py all --no-embedding        (TF-IDF only, no model download)
        python lab6.py all --json results.json --csv retrieval_log.csv
        python lab6.py faq | docbot              (interactive chatbots)
Other sub-commands: tokens, chunks, retrieval, demo, scenarios
"""
import argparse
import csv
import json
import platform
import re
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    sys.stdout.reconfigure(encoding="utf-8")      # Urdu text on Windows consoles
except Exception:
    pass

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_MODEL = {"obj": None, "tried": False, "ok": False, "load_seconds": None}


def get_model():
    """Load the encoder once (CPU). Raises if the package or model is unavailable."""
    if _MODEL["obj"] is None:
        from sentence_transformers import SentenceTransformer
        t = time.perf_counter()
        _MODEL["obj"] = SentenceTransformer(MODEL_NAME, device="cpu")
        _MODEL["load_seconds"] = time.perf_counter() - t
    return _MODEL["obj"]


def embedding_available():
    if not _MODEL["tried"]:
        _MODEL["tried"] = True
        try:
            get_model()
            _MODEL["ok"] = True
        except Exception as e:                      # missing package, no internet, etc.
            print(f"[note] Embedding backend unavailable: {type(e).__name__}: {str(e)[:120]}")
    return _MODEL["ok"]


# =====================================================================================
# Shared retrieval helper (manual: lab6_search.py)
# =====================================================================================
class Retriever:
    def __init__(self, records, backend="tfidf"):
        if not records or any(not r["text"].strip() for r in records):
            raise ValueError("Provide nonempty records")
        self.records, self.backend = records, backend
        texts = [r["text"] for r in records]
        if backend == "tfidf":
            self.vectorizer = TfidfVectorizer()
            self.matrix = self.vectorizer.fit_transform(texts)
        elif backend == "embedding":
            self.model = get_model()
            self.check_length(texts)
            self.matrix = self.model.encode(texts, normalize_embeddings=True)
        else:
            raise ValueError("Use tfidf or embedding")

    def check_length(self, texts):
        for text in texts:
            n = len(self.model.tokenizer(text, truncation=False)["input_ids"])
            if n > self.model.max_seq_length:
                raise ValueError("Input exceeds encoder token limit")

    def search(self, query, k=3, filters=None):
        if not query.strip() or k <= 0:
            return []
        if self.backend == "tfidf":
            q = self.vectorizer.transform([query])
            if q.nnz == 0:                           # no vocabulary overlap -> no matches
                return []
            scores = (q @ self.matrix.T).toarray()[0]
        else:
            self.check_length([query])
            q = self.model.encode([query], normalize_embeddings=True)[0]
            scores = self.matrix @ q
        allowed = [i for i, r in enumerate(self.records)
                   if all(r.get(key) == value for key, value in (filters or {}).items())]
        ranked = sorted(allowed, key=lambda i: (-float(scores[i]), i))
        return [(self.records[i], float(scores[i])) for i in ranked[:k]]


def show_hits(hits):
    if not hits:
        print("No matching records.")
    for record, score in hits:
        print(record["id"], f"score={score:.3f}")
        print(record["text"])


# =====================================================================================
# Chunking with source offsets (manual: task2_chunks.py)
# =====================================================================================
def word_spans(text):
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def windows(text, size=40, overlap=10, offset=0):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    spans = word_spans(text)
    result = []
    for i in range(0, len(spans), size - overlap):
        j = min(i + size, len(spans))
        result.append((offset + spans[i][0], offset + spans[j - 1][1]))
        if j == len(spans):
            break
    return result


def sentence_chunks(text, size=40):
    if size <= 0:
        raise ValueError("size must be positive")
    # Teaching splitter: punctuation followed by whitespace (fails on abbreviations).
    cuts = [0] + [m.end() for m in re.finditer(r"[.!?]\s+", text)] + [len(text)]
    out, pending, count = [], None, 0
    for a, b in zip(cuts, cuts[1:]):
        segment = text[a:b]
        n = len(segment.split())
        if not n:
            continue
        if n > size:                                  # fallback for a very long sentence
            if pending is not None:
                out.append(pending)
            out.extend(windows(segment, size, 0, a))
            pending, count = None, 0
        elif pending is not None and count + n <= size:
            pending, count = (pending[0], b), count + n
        else:
            if pending is not None:
                out.append(pending)
            pending, count = (a, b), n
    if pending is not None:
        out.append(pending)
    return out


def make_chunks(docs, strategy):
    result = []
    for doc_id, text in docs.items():
        spans = sentence_chunks(text) if strategy == "sentence" else windows(text, 40, int(strategy))
        for i, (a, b) in enumerate(spans):
            result.append(dict(id=f"{doc_id}:{i}", doc=doc_id, start=a, end=b, text=text[a:b]))
    return result


STRATEGIES = ["0", "10", "sentence"]
STRATEGY_NAMES = {"0": "window-40 overlap 0", "10": "window-40 overlap 10", "sentence": "sentence-aware 40"}

# =====================================================================================
# Synthetic dataset (manual: create_dataset.py, expanded to 150-250 words per document)
# =====================================================================================
DOCS = {
    "late": (
        "Assessment policy for undergraduate courses. Every assignment has a published deadline, "
        "and the portal closes submissions at that time. Late submission is normally penalized by "
        "ten percent of the available marks for each calendar day, up to a maximum of five days "
        "after which no marks are awarded. Illness extensions require medical evidence before the "
        "deadline. Evidence submitted after the deadline is reviewed only by the examination "
        "committee, and approval is not guaranteed. Students who cannot meet a deadline because of "
        "a personal emergency should contact the course coordinator as early as possible, because "
        "requests that arrive after the work is already late are treated as ordinary late "
        "submissions and are penalized according to the schedule above, even when the student later "
        "provides a convincing explanation of the circumstances. Extensions for travel, work or "
        "other planned activities are not granted. A granted extension never changes the deadline "
        "of other students. Group assignments follow the same rules, and one extension applies to "
        "the whole group only when every member has valid evidence. All dates and penalties in this "
        "document are synthetic teaching data and are not real university policy."),
    "password": (
        "IT help desk guide for student accounts. Every student receives a campus account that "
        "gives access to email, the learning portal and the wireless network. Reset a forgotten "
        "password using the recovery link. The link is sent to the personal email address "
        "registered during admission and expires after thirty minutes, so it should be used "
        "immediately. Never share recovery codes. Staff accounts follow a different process: a "
        "staff member who forgets a password must contact the help desk in person with an identity "
        "card, and the recovery link is not available for staff accounts. If a student has lost "
        "access to the registered email address, the help desk can update it after checking the "
        "student identity card, which normally takes one working day. Accounts are locked after "
        "five wrong password attempts and unlock automatically after fifteen minutes. Passwords "
        "must contain at least ten characters, including a number, and must be changed every year. "
        "The help desk never asks for a password by phone or message, and any request of that kind "
        "should be reported immediately to the security team. All procedures in this guide are "
        "synthetic teaching data."),
    "assignment": (
        "Guide to submitting coursework. Upload assignments through the learning portal. Submit "
        "the final file as a PDF, and keep the file name equal to the roll number followed by the "
        "course code. The portal accepts files up to twenty megabytes and records the exact time of "
        "upload, which is used to decide whether a submission is on time. Email submissions are "
        "accepted only when the portal is unavailable and the coordinator has announced this in "
        "writing, and a screenshot of the error does not replace that announcement. Students should "
        "upload early, because heavy traffic near the deadline can slow the portal and a slow upload "
        "that completes after the deadline counts as late. For problems with the portal, contact the "
        "coordinator, e.g. Dr. Rahman, or the help desk, and attach a short description of the "
        "error. Submissions in other formats, such as word processor files, may be rejected by the "
        "marker. Handwritten work must be scanned into a single PDF in page order. Re-uploading "
        "before the deadline replaces the earlier file, and only the latest version is marked. All "
        "rules here are synthetic teaching data."),
    "contractor": (
        "Workplace policy on remote work. Employees may request remote work with manager approval, "
        "and the approval is reviewed every six months. Contractors can work remotely only if the "
        "signed contract permits it. Employee rules do not apply to contractors. A contractor whose "
        "contract is silent on remote work must attend the office, and any change requires a "
        "written amendment signed by both parties and approved by the human resources office. "
        "Remote workers must be reachable during core hours between ten in the morning and three in "
        "the afternoon, and they must use the company virtual private network when accessing "
        "internal systems. Equipment provided by the company, such as laptops, remains company "
        "property and must be returned when the engagement ends. Remote work from another country "
        "is not allowed without separate approval, because tax and legal rules may differ, and this "
        "restriction applies to employees and contractors alike. Managers should record every "
        "approval in the personnel system so that it can be checked later. This policy is synthetic "
        "teaching data and does not describe any real organisation."),
    "printer": (
        "Troubleshooting guide for office printers and scanners. For blank printer pages, check ink "
        "levels and run the cleaning utility. Print a test page afterwards. If the test page is "
        "still blank, replace the cartridge and repeat the cleaning utility once more, and then "
        "contact the maintenance team. A blank image from a scanner has a different cause: check "
        "document placement and select the correct scanning source, because placing the paper face "
        "up on the glass produces an empty scan on most models. Warning: switch off the printer and "
        "remove the power cable before opening the rear cover, because the heating unit stays hot "
        "for several minutes after use. Paper jams should be cleared by pulling the sheet gently in "
        "the direction of the paper path, never backwards. Printers must not be moved while the "
        "cartridge door is open. Each device has a model label near the power button, and the label "
        "must be quoted in any maintenance request so that the correct manual can be found. All "
        "instructions are synthetic teaching data."),
}
EVIDENCE = {
    "late": "Illness extensions require medical evidence before the deadline.",
    "password": "Reset a forgotten password using the recovery link.",
    "assignment": "Upload assignments through the learning portal.",
    "contractor": "Contractors can work remotely only if the signed contract permits it.",
    "printer": "For blank printer pages, check ink levels and run the cleaning utility.",
}
# Rule + exception/condition pairs that should ideally stay together in ONE chunk.
PAIRS = {
    "late": ("Late submission is normally penalized", EVIDENCE["late"]),
    "password": (EVIDENCE["password"], "the recovery link is not available for staff accounts"),
    "assignment": (EVIDENCE["assignment"], "Email submissions are accepted only when the portal is unavailable"),
    "contractor": (EVIDENCE["contractor"], "Employee rules do not apply to contractors."),
    "printer": (EVIDENCE["printer"], "Print a test page afterwards."),
}


def pair_survival(chunks, docs):
    """Number of documents whose rule AND exception appear together in at least one chunk."""
    return {d: any(PAIRS[d][0] in c["text"] and PAIRS[d][1] in c["text"] for c in chunks if c["doc"] == d)
            for d in docs}


# (query, doc, split). doc None = unanswerable. Evidence labelled BEFORE looking at rankings.
QUESTIONS = [
    ("What is required for illness extensions?", "late", "dev"),
    ("How do I reset a forgotten password?", "password", "dev"),
    ("Where do I upload assignments?", "assignment", "dev"),
    ("Can contractors work remotely?", "contractor", "dev"),
    ("What do I do for blank printer pages?", "printer", "test"),
    ("How can I recover my login access?", "password", "test"),
    ("How should I hand in my coursework?", "assignment", "test"),
    ("What proof is needed when I was ill?", "late", "test"),
    ("What are tomorrow's flight prices?", None, "test"),
    ("Who won the football match?", None, "test"),
]
QUERIES = [dict(query=q, doc=d, evidence=EVIDENCE[d] if d else None, split=s) for q, d, s in QUESTIONS]


def check_dataset():
    for k, t in DOCS.items():
        n = len(t.split())
        assert 150 <= n <= 250, f"{k}: {n} words"
        assert EVIDENCE[k] in t, f"evidence of {k} is not an exact substring"
        assert all(p in t for p in PAIRS[k]), f"pair of {k} is not an exact substring"
    return {k: len(t.split()) for k, t in DOCS.items()}


# =====================================================================================
# Metrics (manual: guided_metrics.py)
# =====================================================================================
def retrieval_metrics(rankings, relevant):
    hits, rr = [], []
    for ids, good in zip(rankings, relevant):
        ranks = [r for r, chunk_id in enumerate(ids[:3], 1) if chunk_id in good]
        hits.append(bool(ranks))
        rr.append(1 / ranks[0] if ranks else 0)
    return sum(hits) / len(hits), sum(rr) / len(rr)


def evaluate(scores, chunks, items):
    """Hit@3 and MRR@3 over ANSWERABLE items. Relevant = same doc AND chunk holds the full evidence."""
    hits, reciprocal = [], []
    for row, q in zip(scores, items):
        if q["doc"] is None:
            continue
        top = np.argsort(-row, kind="stable")[:3]
        good = [rank for rank, i in enumerate(top, 1)
                if chunks[i]["doc"] == q["doc"] and q["evidence"] in chunks[i]["text"]]
        hits.append(bool(good))
        reciprocal.append(1 / good[0] if good else 0)
    return (float(np.mean(hits)), float(np.mean(reciprocal))) if hits else (None, None)


# =====================================================================================
# Task 1: tokenization and FAQ chatbot
# =====================================================================================
TOKEN_SAMPLES = ["AI, search!", "I can't log in.", "help@example.com", "CS4944", "",
                 "یہ ایک مثال ہے"]


def task1_tokens(R):
    print("=== Task 1: whitespace vs word-like tokens ===")
    rows = []
    for text in TOKEN_SAMPLES:
        ws = text.split()
        words = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
        print(f"Original : {text!r}")
        print(f"  whitespace: {ws} ({len(ws)})")
        print(f"  word-like : {words} ({len(words)})")
        rows.append(dict(text=text, whitespace=ws, words=words, model=None))
    assert re.findall(r"\b\w+\b", "") == []
    if embedding_available():
        model = get_model()
        print("Model sub-word tokens (all-MiniLM-L6-v2):")
        for row in rows[:2]:
            row["model"] = model.tokenizer.tokenize(row["text"])
            print(f"  {row['text']!r}: {row['model']} ({len(row['model'])})")
    else:
        print("Model sub-word tokens: skipped (encoder not available).")
    R["tokens"] = rows


FAQ = [
    {"id": "faq:password", "question": "How do I reset my password?",
     "answer": "Use the recovery link on the login page.", "source": "it_help.txt"},
    {"id": "faq:submit", "question": "Where do I submit my assignment?",
     "answer": "Use the assignment upload page in the learning portal.", "source": "assessment.txt"},
    {"id": "faq:late", "question": "Can illness justify a late submission?",
     "answer": "Request an extension with medical evidence before the deadline.", "source": "handbook.txt"},
    # two FAQs added for Task 1 (each with a source)
    {"id": "faq:printer", "question": "Why does my printer print blank pages?",
     "answer": "Check ink levels and run the cleaning utility.", "source": "printer_guide.txt"},
    {"id": "faq:contractor", "question": "Can contractors work remotely?",
     "answer": "Only if the signed contract permits it.", "source": "hr_policy.txt"},
]


def build_bot(backend="tfidf"):
    records = [dict(r, text=r["question"] + " " + r["answer"]) for r in FAQ]
    return Retriever(records, backend)


def faq_reply(engine, query):
    if not query.strip():
        return "Please enter a question."
    # Only exact canonical FAQ text is answered automatically.
    for r in FAQ:
        if query.strip().casefold() == r["question"].casefold():
            return r["answer"] + " [source: " + r["source"] + "]"
    hits = engine.search(query)
    if not hits:
        return "No matching FAQ. Please contact the help desk."
    lines = ["Possible matches; choose or rephrase your question:"]
    for r, score in hits:
        lines.append(r["question"] + f" (similarity {score:.3f})")
    return "\n".join(lines)


def faq_demo(R):
    print("\n=== FAQ chatbot transcript (TF-IDF) ===")
    engine = build_bot()
    transcript = []
    for q in ["How do I reset my password?", "password recovery", "Why is my printer blank?",
              "", "volcano geology"]:
        a = faq_reply(engine, q)
        print("You:", q)
        print("Bot:", a)
        transcript.append(dict(query=q, reply=a))
    # checks from the reference
    assert "it_help.txt" in faq_reply(engine, "How do I reset my password?")
    assert faq_reply(engine, "") == "Please enter a question."
    assert "No matching FAQ" in faq_reply(engine, "volcano geology")
    assert "Possible matches" in faq_reply(engine, "password recovery")
    print("FAQ behaviour checks passed")
    R["faq"] = transcript
    # 6.7 suggestions
    print("\nSuggested questions for 'assignment upload':")
    sug = []
    for record, score in engine.search("assignment upload", k=3):
        if score > 0:
            print(record["question"], f"({score:.3f})")
            sug.append(dict(question=record["question"], score=score))
    R["suggestions"] = sug


def faq_interactive(backend):
    engine = build_bot(backend)
    print("FAQ assistant. Type exit to stop.")
    while True:
        try:
            query = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() == "exit":
            break
        print("Bot:", faq_reply(engine, query))


# =====================================================================================
# Task 2: chunking and semantic suggestions
# =====================================================================================
TASK2_DOCS = {k: DOCS[k] for k in ["late", "password", "assignment"]}


def task2_chunks(R):
    print("\n=== Task 2: chunk boundary tests ===")
    tests = []

    def record(name, ok):
        tests.append(dict(name=name, passed=bool(ok)))
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    record("empty text gives no windows", windows("") == [])
    record("short document gives one window", windows("short text") == [(0, 10)])
    try:
        windows("a b", 2, 2)
        record("invalid overlap raises ValueError", False)
    except ValueError:
        record("invalid overlap raises ValueError", True)
    long = " ".join(["word"] * 95) + "."
    parts = sentence_chunks(long)
    record("95-word sentence splits into 3 fallback windows (each <= 40 words)",
           len(parts) == 3 and all(len(long[a:b].split()) <= 40 for a, b in parts))
    tiny = {"example": "Reset your password. Keep recovery codes private."}
    for s in STRATEGIES:
        for c in make_chunks(tiny, s):
            assert c["text"] == tiny[c["doc"]][c["start"]:c["end"]]
    record("chunk text equals source[start:end] for every strategy", True)
    abbr = "Contact Dr. Rahman first. He handles portal problems."
    cut = [abbr[a:b] for a, b in sentence_chunks(abbr, size=4)]
    print("  Abbreviation example, size 4:", cut)
    R["task2_tests"] = tests

    print("\n=== Task 2: chunking comparison (3 documents) ===")
    original_words = sum(len(t.split()) for t in TASK2_DOCS.values())
    stats = []
    for s in STRATEGIES:
        chunks = make_chunks(TASK2_DOCS, s)
        sizes = [len(c["text"].split()) for c in chunks]
        emitted = sum(sizes)
        survive = {d: any(EVIDENCE[d] in c["text"] for c in chunks if c["doc"] == d) for d in TASK2_DOCS}
        pairs_ok = pair_survival(chunks, TASK2_DOCS)
        row = dict(pair_survives=pairs_ok, pairs_kept=sum(pairs_ok.values()), strategy=s, name=STRATEGY_NAMES[s], chunks=len(chunks), mean_words=float(np.mean(sizes)),
                   min_words=min(sizes), max_words=max(sizes), duplicate_words=emitted - original_words,
                   evidence_survives=survive, survived=sum(survive.values()), of=len(survive))
        stats.append(row)
        print(f"{row['name']:22s} chunks={row['chunks']:2d} mean={row['mean_words']:.1f} words "
              f"min={row['min_words']} max={row['max_words']} duplicate_words={row['duplicate_words']} "
              f"evidence in one chunk: {row['survived']}/{row['of']}; rule+exception together: {row['pairs_kept']}/{row['of']} {pairs_ok}")
    R["task2_stats"] = stats
    # show one split rule
    for s in ["0", "10"]:
        chunks = make_chunks(TASK2_DOCS, s)
        for d in TASK2_DOCS:
            if not any(EVIDENCE[d] in c["text"] for c in chunks if c["doc"] == d):
                ends = [(c["id"], c["start"], c["end"]) for c in chunks if c["doc"] == d]
                print(f"  Evidence split across chunks ({STRATEGY_NAMES[s]}, doc '{d}'): {ends}")

    print("\n=== Task 2: paraphrased queries, sentence-aware chunks ===")
    chunks = make_chunks(TASK2_DOCS, "sentence")
    paraphrases = [("How can I regain access to my account?", "password"),
                   ("What proof is needed when I was ill?", "late"),
                   ("How should I hand in my coursework?", "assignment")]
    sem = []
    backends = ["tfidf"] + (["embedding"] if R["env"]["embedding_available"] else [])
    for backend in backends:
        engine = Retriever(chunks, backend)
        for q, doc in paraphrases:
            hits = engine.search(q, k=3)
            rows = [dict(id=r["id"], score=sc, start=r["start"], end=r["end"],
                         has_evidence=(r["doc"] == doc and EVIDENCE[doc] in r["text"])) for r, sc in hits]
            print(f"[{backend}] {q!r} -> " + (", ".join(f"{x['id']} {x['score']:.3f}{' (evidence)' if x['has_evidence'] else ''}" for x in rows) or "no matches"))
            sem.append(dict(backend=backend, query=q, target=doc, top3=rows))
    R["task2_semantic"] = sem


# =====================================================================================
# Task 3: evaluation of retrieval strategies
# =====================================================================================
def task3_retrieval(R, csv_path=None):
    print("\n=== Task 3: retrieval evaluation (5 documents, 8 answerable + 2 unanswerable queries) ===")
    docs, items = DOCS, QUERIES
    query_texts = [q["query"] for q in items]
    use_emb = R["env"]["embedding_available"]
    model = get_model() if use_emb else None
    load_s = _MODEL["load_seconds"]
    if use_emb:
        print(f"Model load seconds (measured separately): {load_s:.2f} | max_seq_length = {model.max_seq_length}")
    methods = ["tfidf"] + (["embedding"] if use_emb else [])
    metrics, logs, failures, unanswerable, audit = [], [], [], [], []
    pair_counts = {}
    for strategy in STRATEGIES:
        chunks = make_chunks(docs, strategy)
        texts = [c["text"] for c in chunks]
        pair_counts[strategy] = sum(pair_survival(chunks, docs).values())
        print(f"{STRATEGY_NAMES[strategy]}: rule+exception kept together in {pair_counts[strategy]}/{len(docs)} documents")
        if use_emb:
            lengths = [len(model.tokenizer(t, truncation=False)["input_ids"]) for t in texts + query_texts]
            audit.append(dict(strategy=strategy, max_tokens=max(lengths), limit=model.max_seq_length,
                              fits=max(lengths) <= model.max_seq_length))
            print(f"Token audit {STRATEGY_NAMES[strategy]}: max {max(lengths)} tokens, limit {model.max_seq_length}")
            if max(lengths) > model.max_seq_length:
                raise ValueError("Model limit exceeded; rechunk or shorten query")
        for method in methods:
            t = time.perf_counter()
            if method == "tfidf":
                vectorizer = TfidfVectorizer()
                C = vectorizer.fit_transform(texts)
                Q = vectorizer.transform(query_texts)
            else:
                C = model.encode(texts, normalize_embeddings=True)
                Q = model.encode(query_texts, normalize_embeddings=True)
            encode_s = time.perf_counter() - t
            t = time.perf_counter()
            scores = (Q @ C.T).toarray() if method == "tfidf" else Q @ C.T
            search_s = time.perf_counter() - t
            for split in ["dev", "test"]:
                idx = [i for i, q in enumerate(items) if q["split"] == split]
                hit, mrr = evaluate(scores[idx], chunks, [items[i] for i in idx])
                metrics.append(dict(strategy=strategy, method=method, split=split, hit=hit, mrr=mrr,
                                    chunks=len(chunks), encode_s=encode_s, search_s=search_s))
                print(f"{STRATEGY_NAMES[strategy]:22s} {method:9s} {split:4s} Hit@3={hit:.3f} MRR@3={mrr:.3f} "
                      f"chunks={len(chunks)} encode={encode_s:.4f}s search={search_s:.4f}s")
            for row, q in zip(scores, items):
                top = np.argsort(-row, kind="stable")[:3]
                for rank, i in enumerate(top, 1):
                    c = chunks[i]
                    logs.append(dict(strategy=strategy, method=method, query=q["query"], split=q["split"],
                                     rank=rank, score=float(row[i]), chunk_id=c["id"], doc=c["doc"],
                                     start=c["start"], end=c["end"], text=c["text"],
                                     answerable=q["doc"] is not None))
                if q["doc"] is None:
                    c = chunks[top[0]]
                    unanswerable.append(dict(strategy=strategy, method=method, query=q["query"],
                                             top1=c["id"], score=float(row[top[0]]), text=c["text"]))
                else:
                    good = [r for r, i in enumerate(top, 1)
                            if chunks[i]["doc"] == q["doc"] and q["evidence"] in chunks[i]["text"]]
                    if not good:
                        c = chunks[top[0]]
                        failures.append(dict(strategy=strategy, method=method, query=q["query"], split=q["split"],
                                             expected_doc=q["doc"], top1=c["id"], score=float(row[top[0]]),
                                             top1_text=c["text"],
                                             evidence_chunks=[x["id"] for x in chunks if x["doc"] == q["doc"]
                                                              and q["evidence"] in x["text"]]))
    # strategy selection on DEV only, then report held-out TEST without retuning
    selected = {}
    for method in methods:
        dev = [m for m in metrics if m["method"] == method and m["split"] == "dev"]
        # order: dev MRR, dev Hit@3, then rule+exception kept together (stated before testing), then fewer chunks
        best = max(dev, key=lambda m: (m["mrr"], m["hit"], pair_counts[m["strategy"]], -m["chunks"]))
        test = [m for m in metrics if m["method"] == method and m["split"] == "test"
                and m["strategy"] == best["strategy"]][0]
        selected[method] = dict(strategy=best["strategy"], dev_hit=best["hit"], dev_mrr=best["mrr"],
                                test_hit=test["hit"], test_mrr=test["mrr"])
        print(f"Selected on dev ({method}): {STRATEGY_NAMES[best['strategy']]} "
              f"(dev MRR {best['mrr']:.3f}) -> held-out test Hit@3={test['hit']:.3f} MRR@3={test['mrr']:.3f}")
    print(f"\nFailures (answerable query, no relevant chunk in top 3): {len(failures)}")
    for f in failures[:12]:
        print(f"  [{f['method']}, {STRATEGY_NAMES[f['strategy']]}] {f['query']!r} -> top1 {f['top1']} "
              f"({f['score']:.3f}); chunks that hold the evidence: {f['evidence_chunks'] or 'NONE (evidence split)'}")
    print("\nUnanswerable queries still return text (top-1 per configuration):")
    for u in unanswerable:
        print(f"  [{u['method']}, {STRATEGY_NAMES[u['strategy']]}] {u['query']!r} -> {u['top1']} score={u['score']:.3f}")
    # dev calibration hint: top-1 score of correct answerable dev queries vs unanswerable
    R["task3"] = dict(pair_counts=pair_counts, model_load_s=load_s, methods=methods, metrics=metrics, selected=selected,
                      failures=failures, unanswerable=unanswerable, token_audit=audit)
    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(logs[0]))
            writer.writeheader()
            writer.writerows(logs)
        print("Wrote", csv_path)
    return selected


# =====================================================================================
# Document chatbot (confirmation-based evidence display)
# =====================================================================================
def doc_reply(engine, query, choice):
    """Return printable lines. `choice` is the user's selection ('' = none)."""
    out = []
    hits = engine.search(query)
    if not hits:
        return ["No candidate evidence; rephrase or contact a human."]
    out.append("Candidate evidence; relevance must be checked:")
    for number, (r, score) in enumerate(hits, 1):
        out.append(f"{number} {r['id']} {score:.3f} characters {r['start']} {r['end']}")
    if not str(choice).isdigit() or not 1 <= int(choice) <= len(hits):
        out.append("No answer selected. Rephrase or contact a human.")
        return out
    r, score = hits[int(choice) - 1]
    out.append("Retrieved passage: " + r["text"])
    out.append(f"Source: {r['doc']} characters {r['start']} {r['end']}")
    return out


def doc_demo(R, strategy, backend):
    print(f"\n=== Document chatbot transcript ({STRATEGY_NAMES[strategy]}, {backend}) ===")
    engine = Retriever(make_chunks(DOCS, strategy), backend)
    transcript = []
    for q, choice in [("Can contractors work remotely?", "1"), ("What proof is needed when I was ill?", "1"),
                      ("", ""), ("What are tomorrow's flight prices?", "")]:
        lines = doc_reply(engine, q, choice)
        print("Question:", repr(q), "| choice:", repr(choice))
        for ln in lines:
            print("  " + ln)
        transcript.append(dict(query=q, choice=choice, lines=lines))
    R["docbot"] = dict(strategy=strategy, backend=backend, transcript=transcript)


def doc_interactive(strategy, backend):
    engine = Retriever(make_chunks(DOCS, strategy), backend)
    print("Evidence assistant. Type exit to stop.")
    while True:
        try:
            query = input("Question: ")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() == "exit":
            break
        hits = engine.search(query)
        if not hits:
            print("No candidate evidence; rephrase or contact a human.")
            continue
        choice = ""
        print("Candidate evidence; relevance must be checked:")
        for number, (r, score) in enumerate(hits, 1):
            print(number, r["id"], f"{score:.3f}", "characters", r["start"], r["end"])
        choice = input("Choose 1-3 to inspect; Enter if none apply: ")
        for ln in doc_reply(engine, query, choice)[len(hits) + 1:]:
            print(ln)


# =====================================================================================
# Scenarios 1-5 (TF-IDF, metadata filters) and metric self-check
# =====================================================================================
def scenarios(R):
    print("\n=== Scenarios 1-5 (TF-IDF with metadata filters) ===")
    cases = [
        ("1 handbook", [
            {"id": "assessment:late", "source": "handbook.txt",
             "text": "Late submission is normally penalized. Students with illness may request an extension "
                     "with medical evidence before the deadline."},
            {"id": "assessment:format", "source": "handbook.txt",
             "text": "Submit the assignment as a PDF through the learning portal."}],
         "illness medical evidence late submission", None, "assessment:late"),
        ("2 support", [
            {"id": "returns:damage", "region": "PK",
             "text": "For a damaged parcel, contact support within 48 hours and attach photographs of the item and packaging."},
            {"id": "returns:change", "region": "PK", "text": "For a change of mind, return unused items within 14 days."},
            {"id": "returns:damage:other", "region": "OTHER",
             "text": "Damaged parcel reports in the other region have a 72 hour window."}],
         "damaged parcel photographs", {"region": "PK"}, "returns:damage"),
        ("3 docs", [
            {"id": "auth:v2", "version": "v2",
             "text": "Renew an expired API token with POST /v2/token/refresh. Send the refresh_token field; never print its value in logs."},
            {"id": "auth:v1", "version": "v1", "text": "Renew an expired API token with POST /v1/auth/renew."}],
         "renew expired API token", {"version": "v2"}, "auth:v2"),
        ("4 workplace", [
            {"id": "remote:staff", "audience": "employee", "text": "Employees may request remote work with manager approval."},
            {"id": "remote:contractor", "audience": "contractor",
             "text": "Contractors may work remotely only when the signed contract explicitly permits remote work."}],
         "remote work", {"audience": "contractor"}, "remote:contractor"),
        ("5 equipment", [
            {"id": "printer:blank", "device": "printer",
             "text": "For blank printer pages: 1. Check ink level. 2. Run the printer cleaning utility. "
                     "3. Print a test page. Follow the device manual."},
            {"id": "scanner:blank", "device": "scanner",
             "text": "For a blank scanner image, check document placement and select the correct scanning source."}],
         "blank pages", {"device": "printer"}, "printer:blank"),
    ]
    out = []
    for name, records, query, filters, expected in cases:
        hits = Retriever(records).search(query, k=1, filters=filters)
        got = hits[0][0]["id"] if hits else None
        print(f"Scenario {name}: query {query!r} filters {filters} -> {got} ({'PASS' if got == expected else 'FAIL'})")
        out.append(dict(scenario=name, query=query, filters=filters, result=got, expected=expected))
    # nonexistent version returns no records
    none = Retriever(cases[2][1]).search("renew expired API token", k=1, filters={"version": "v9"})
    print("Scenario 3 nonexistent version v9 ->", none)
    hit, mrr = retrieval_metrics([["a", "b", "c"], ["c", "a", "b"], ["x", "y", "z"]], [{"a"}, {"a"}, {"a"}])
    assert abs(hit - 2 / 3) < 1e-9 and abs(mrr - 0.5) < 1e-9
    print(f"Metric self-check: Hit@3={hit:.3f}; MRR@3={mrr:.3f} (expected 0.667 / 0.500)")
    R["scenarios"] = out


# =====================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", default="all",
                    choices=["all", "tokens", "chunks", "retrieval", "faq", "demo", "docbot", "scenarios"])
    ap.add_argument("--no-embedding", action="store_true", help="skip the sentence-embedding backend")
    ap.add_argument("--backend", choices=["tfidf", "embedding"], default="tfidf")
    ap.add_argument("--json", help="write structured results to this file")
    ap.add_argument("--csv", help="write the top-3 retrieval log to this CSV file")
    args = ap.parse_args()

    if args.command == "faq":
        return faq_interactive(args.backend)
    R = {"env": {"python": platform.python_version(), "platform": platform.platform(), "model": MODEL_NAME}}
    R["env"]["embedding_available"] = (not args.no_embedding) and embedding_available()
    try:
        import sklearn
        R["env"]["sklearn"], R["env"]["numpy"] = sklearn.__version__, np.__version__
        if R["env"]["embedding_available"]:
            import sentence_transformers
            R["env"]["sentence_transformers"] = sentence_transformers.__version__
    except Exception:
        pass
    R["doc_words"] = check_dataset()
    print("Documents (words):", R["doc_words"], "| embedding backend:", R["env"]["embedding_available"])

    if args.command == "docbot":
        sel = "sentence"
        return doc_interactive(sel, "embedding" if R["env"]["embedding_available"] else "tfidf")
    if args.command in ("all", "tokens"):
        task1_tokens(R)
    if args.command in ("all", "demo"):
        faq_demo(R)
    if args.command in ("all", "chunks"):
        task2_chunks(R)
    if args.command in ("all", "retrieval"):
        selected = task3_retrieval(R, args.csv)
        method = "embedding" if R["env"]["embedding_available"] else "tfidf"
        doc_demo(R, selected[method]["strategy"], method)
    if args.command in ("all", "scenarios"):
        scenarios(R)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(R, f, ensure_ascii=False, indent=1)
        print("Wrote", args.json)


if __name__ == "__main__":
    main()
