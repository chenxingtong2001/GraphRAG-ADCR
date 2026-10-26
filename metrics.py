def unique_ordered(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def precision_recall_f1(pred, gold):
    (pred, gold) = (set(pred), set(gold))
    if not pred or not gold:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    inter = len(pred & gold)
    if inter == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    p = inter / len(pred)
    r = inter / len(gold)
    f1 = 2 * p * r / (p + r)
    return {'precision': round(p, 4), 'recall': round(r, 4), 'f1': round(f1, 4)}

def precision_at_k(ranked, relevant, k):
    relevant = set(relevant)
    if k <= 0 or not ranked:
        return 0.0
    topk = unique_ordered(ranked)[:k]
    hits = sum((1 for x in topk if x in relevant))
    return round(hits / k, 4)

def recall_at_k(ranked, relevant, k):
    relevant = set(relevant)
    if k <= 0 or not relevant:
        return 0.0
    topk = unique_ordered(ranked)[:k]
    hits = sum((1 for x in topk if x in relevant))
    return round(hits / len(relevant), 4)

def mrr(ranked, relevant):
    relevant = set(relevant)
    for (i, x) in enumerate(unique_ordered(ranked), start=1):
        if x in relevant:
            return round(1.0 / i, 4)
    return 0.0

def _preprocess_zh(text):
    out = []
    for ch in text:
        if '一' <= ch <= '\u9fff':
            out.append(' ' + ch + ' ')
        else:
            out.append(ch)
    return ''.join(out)

def _make_cjk_scorer():
    from rouge_score import rouge_scorer
    from rouge_score.rouge_scorer import tokenizers

    class CJKTokenizer(tokenizers.Tokenizer):

        def tokenize(self, text):
            return [t for t in text.split(' ') if t]
    return rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False, tokenizer=CJKTokenizer())

def rouge_scores(answer, reference):
    if not answer or not reference:
        return {}
    try:
        scorer = _make_cjk_scorer()
    except ImportError:
        return {}
    scores = scorer.score(_preprocess_zh(reference), _preprocess_zh(answer))
    return {'rouge-1': round(scores['rouge1'].fmeasure, 4), 'rouge-2': round(scores['rouge2'].fmeasure, 4), 'rouge-l': round(scores['rougeL'].fmeasure, 4)}
