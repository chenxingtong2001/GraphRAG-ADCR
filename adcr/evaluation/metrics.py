# -*- coding: utf-8 -*-
"""Set precision/recall/F1, character-level ROUGE, and ranked retrieval metrics."""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def unique_ordered(seq):
    """去重并保持原有顺序。"""
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def precision_recall_f1(pred, gold):
    """集合级 P/R/F1。pred、gold 均为可迭代集合；空集边界返回 0。"""
    pred, gold = set(pred), set(gold)
    if not pred or not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    inter = len(pred & gold)
    if inter == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    p = inter / len(pred)
    r = inter / len(gold)
    f1 = 2 * p * r / (p + r)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def precision_at_k(ranked, relevant, k):
    """前 k 个结果中相关项占比。"""
    relevant = set(relevant)
    if k <= 0 or not ranked:
        return 0.0
    topk = unique_ordered(ranked)[:k]
    hits = sum(1 for x in topk if x in relevant)
    return round(hits / k, 4)


def recall_at_k(ranked, relevant, k):
    """相关项被前 k 个结果覆盖的比例。"""
    relevant = set(relevant)
    if k <= 0 or not relevant:
        return 0.0
    topk = unique_ordered(ranked)[:k]
    hits = sum(1 for x in topk if x in relevant)
    return round(hits / len(relevant), 4)


def mrr(ranked, relevant):
    """平均倒数排名：第一个相关结果的排名倒数。"""
    relevant = set(relevant)
    for i, x in enumerate(unique_ordered(ranked), start=1):
        if x in relevant:
            return round(1.0 / i, 4)
    return 0.0


def _preprocess_zh(text):
    """中文字符两侧插空格，使按空格分词的 tokenizer 得到字符级 token。"""
    out = []
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            out.append(" " + ch + " ")
        else:
            out.append(ch)
    return "".join(out)


def _make_cjk_scorer():
    """构造中文友好的 RougeScorer：自定义 tokenizer 绕过默认 ASCII 清洗。"""
    from rouge_score import rouge_scorer
    from rouge_score.rouge_scorer import tokenizers

    class CJKTokenizer(tokenizers.Tokenizer):
        def tokenize(self, text):
            return [t for t in text.split(" ") if t]

    return rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
        tokenizer=CJKTokenizer(),
    )


def rouge_scores(answer, reference):
    """ROUGE-1/2/L（fmeasure，中文按字符级计算）。未装 rouge-score 时返回空 dict。"""
    if not answer or not reference:
        return {}
    try:
        scorer = _make_cjk_scorer()
    except ImportError:
        return {}
    # 默认 tokenizer 会清空非 ASCII 字符；先做字符级空格预处理再送自定义 tokenizer
    scores = scorer.score(_preprocess_zh(reference), _preprocess_zh(answer))
    return {
        "rouge-1": round(scores["rouge1"].fmeasure, 4),
        "rouge-2": round(scores["rouge2"].fmeasure, 4),
        "rouge-l": round(scores["rougeL"].fmeasure, 4),
    }


# ---------------- 离线自测 ----------------
def _selftest():
    checks = []

    def check(name, got, expect, tol=1e-6):
        ok = abs(got - expect) < tol if isinstance(expect, float) else got == expect
        checks.append((name, ok, got, expect))

    # P/R/F1：pred={a,b,c}, gold={b,c,d} -> P=2/3, R=2/3, F1=2/3
    m = precision_recall_f1({"a", "b", "c"}, {"b", "c", "d"})
    check("precision", m["precision"], round(2 / 3, 4))
    check("recall", m["recall"], round(2 / 3, 4))
    check("f1", m["f1"], round(2 / 3, 4))
    # 空集边界
    check("empty-pred", precision_recall_f1(set(), {"a"})["f1"], 0.0)
    check("empty-gold", precision_recall_f1({"a"}, set())["f1"], 0.0)
    # 完全一致
    check("identical", precision_recall_f1({"a"}, {"a"})["f1"], 1.0)

    # 检索指标：ranked=[c1..c5], relevant={c2,c4}
    ranked = ["c1", "c2", "c3", "c4", "c5"]
    relevant = {"c2", "c4"}
    check("p@3", precision_at_k(ranked, relevant, 3), 0.3333)
    check("r@3", recall_at_k(ranked, relevant, 3), 0.5)
    check("p@2", precision_at_k(ranked, relevant, 2), 0.5)
    check("r@2", recall_at_k(ranked, relevant, 2), 0.5)
    check("mrr-hit2", mrr(["c1", "c2", "c3"], {"c2"}), 0.5)
    check("mrr-hit1", mrr(["c1", "c2"], {"c1"}), 1.0)
    check("mrr-miss", mrr(["c1", "c2"], {"c9"}), 0.0)
    # 去重保序：重复结果不影响排名
    check("mrr-dedup", mrr(["c1", "c1", "c2"], {"c2"}), 0.5)

    # ROUGE：相同文本应为 1.0；不同文本应小于 1
    r_same = rouge_scores("张三承担全部责任", "张三承担全部责任")
    if r_same:
        check("rouge-1-same", r_same.get("rouge-1", 0), 1.0)
        r_diff = rouge_scores("李四承担次要责任", "张三承担全部责任")
        check("rouge-1-diff<1", 1.0 if r_diff.get("rouge-1", 1) < 1.0 else 0.0, 1.0)
    else:
        print("（rouge-score 未安装，ROUGE 自测跳过）")

    failed = [c for c in checks if not c[1]]
    for name, ok, got, expect in checks:
        print("%s %-16s got=%s expect=%s" % ("PASS" if ok else "FAIL", name, got, expect))
    print("\n自测结果：%d / %d 通过" % (len(checks) - len(failed), len(checks)))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    _selftest()
