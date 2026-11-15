# -*- coding: utf-8 -*-
"""BM25 and dense semantic retrieval with reciprocal rank fusion."""
import argparse
import collections
import json
import math
import re

from adcr import config
from adcr.retrieval.embeddings import semantic_scores, settings as embedding_settings
from adcr.graph.annotations import load_cases, load_laws
from adcr.graph.build_graph import dataset_key, get_driver
from adcr.graph.schema import validate_case
from adcr.common.storage import fingerprint, read_json, write_json

RETRIEVAL_VERSION = 'hybrid-bm25-dense-rrf'
STOP = {'责任', '认定', '因果', '关系', '案件', '如何', '应当', '承担', '纠纷', '问题', '本案'}


class RetrievalScoringError(RuntimeError):
    """Scoring failures must not be mistaken for a Neo4j connectivity failure."""


def tokenize(text):
    result = re.findall(r'[a-z0-9]+', text.lower())
    for run in re.findall(r'[\u4e00-\u9fff]+', text):
        result.extend(run[i:i+2] for i in range(len(run)-1) if run[i:i+2] not in STOP)
    return result


def _bm25(query, documents):
    counters = [collections.Counter(tokenize(d)) for d in documents]
    lengths = [sum(c.values()) for c in counters]
    avg = sum(lengths) / max(len(lengths), 1) or 1
    terms = set(tokenize(query))
    df = {t: sum(t in c for c in counters) for t in terms}
    scores = []
    for counter, length in zip(counters, lengths):
        score = 0.0
        for term in terms:
            tf = counter[term]
            if tf:
                idf = math.log(1 + (len(counters) - df[term] + .5) / (df[term] + .5))
                score += idf * tf * 2.5 / (tf + 1.5 * (.25 + .75 * length / avg))
        scores.append(score)
    return scores


def _query_text(case):
    return '\n'.join(case.get(k, '') for k in ('facts', 'question', 'case_type'))


def _path_text(path):
    return ' '.join(path['chain'] + path.get('law_articles', []) +
                    [path['explanation']] + path.get('evidence_quotes', []))


def _document(data):
    return ' '.join([data['case_name'], data['case_type'], data.get('source_text', '')] +
                    [e['name'] + ' ' + e.get('desc', '') for e in data['entities']])


def load_extracted():
    return load_cases()


def retrieval_settings():
    weights = (config.HYBRID_KEYWORD_WEIGHT, config.HYBRID_SEMANTIC_WEIGHT)
    if any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError('混合检索两路权重必须为有限正数')
    if config.HYBRID_RRF_K < 1 or config.HYBRID_CANDIDATES < 1:
        raise ValueError('RRF 常数及候选数量必须为正整数')
    return dict(method=RETRIEVAL_VERSION, embedding=embedding_settings(),
                keyword_weight=weights[0], semantic_weight=weights[1],
                rrf_k=config.HYBRID_RRF_K, candidates=config.HYBRID_CANDIDATES)


def _hybrid(query, documents, ids, limit):
    """Union of two top-N lists, fused without mixing incompatible score scales."""
    options = retrieval_settings()
    lexical = _bm25(query, documents)
    try:
        dense = semantic_scores(query, documents)
    except Exception as exc:
        raise RetrievalScoringError('语义检索失败：%s' % exc) from exc
    if len(dense) != len(documents) or any(not math.isfinite(s) for s in dense):
        raise RetrievalScoringError('语义检索分数数量或数值无效')
    keyword_order = sorted((i for i,s in enumerate(lexical) if s > 0),
                           key=lambda i: (-lexical[i], ids[i]))[:limit]
    semantic_order = sorted(range(len(documents)), key=lambda i: (-dense[i], ids[i]))[:limit]
    kr = {i:r for r,i in enumerate(keyword_order, 1)}
    sr = {i:r for r,i in enumerate(semantic_order, 1)}
    result = []
    for i in sorted(set(kr) | set(sr)):
        score = sum(weight / (options['rrf_k'] + ranks[i])
                    for weight,ranks in ((options['keyword_weight'],kr),
                                         (options['semantic_weight'],sr)) if i in ranks)
        result.append((i, dict(bm25_score=round(lexical[i], 6),
                              semantic_score=round(dense[i], 6),
                              keyword_rank=kr.get(i), semantic_rank=sr.get(i),
                              fusion_score=score)))
    return sorted(result, key=lambda row: (-row[1]['fusion_score'], ids[row[0]]))


def _rank(case, candidates, top_k, laws, paths_per_case):
    query = _query_text(case)
    if not query.strip():
        raise ValueError('检索查询不能为空')
    qt = set(tokenize(query))
    excluded = set(case.get('exclude_case_ids', [])) | {case.get('source_case_id', '')}
    # Exclude self/related judgments before computing either ranking.
    ordered = [(cid,d) for cid,d in sorted(candidates.items()) if cid not in excluded
               and not (case.get('event_group') and d.get('event_group') == case['event_group'])
               and d['causal_paths']]
    fused = _hybrid(query, [_document(d) for _,d in ordered],
                    [cid for cid,_ in ordered], max(top_k, config.HYBRID_CANDIDATES))
    ranked, paths = [], []
    for rank, (index, signals) in enumerate(fused[:top_k], 1):
        cid, data = ordered[index]
        score = round(signals['fusion_score'], 8)
        matched = sorted(qt & set(tokenize(_document(data))))
        ranked.append(dict(case_id=cid, case_name=data['case_name'], rank=rank,
                           similarity=score, matched_terms=matched[:30], **signals,
                           annotation_status=data['_annotation_status']))
        cps = data['causal_paths']
        path_order = _hybrid(query, [_path_text(p) for p in cps],
                             [p['path_id'] for p in cps], len(cps))
        for pi, path_signals in path_order[:paths_per_case]:
            path = cps[pi]
            paths.append(dict(path, case_id=cid, case_name=data['case_name'], rank=rank,
                              similarity=score, path_similarity=round(path_signals['fusion_score'], 8),
                              retrieval_scores=path_signals,
                              annotation_status=data['_annotation_status'],
                              source_file=data.get('_source_file', '')))
    cited = {name for p in paths for name in p.get('law_articles', [])}
    resources = []
    for law in laws:
        overlap = len(qt & set(tokenize(law['name'] + law['text'])))
        if overlap or law['name'] in cited:
            resources.append((overlap, law))
    resources.sort(key=lambda row: (-row[0], row[1]['id']))
    return dict(ranked_cases=ranked, paths=paths, laws=[law for _, law in resources[:3]],
                retrieval_config=retrieval_settings())


def retrieve_memory(case, extracted, top_k=3, laws=None, paths_per_case=2):
    return _rank(case, extracted, top_k, laws or [], paths_per_case)


def retrieve_neo4j(case, extracted, top_k=3, laws=None, paths_per_case=2):
    laws = laws or []
    dataset = dataset_key(extracted, laws)
    candidates = {}
    with get_driver() as driver:
        driver.verify_connectivity()
        with driver.session(database=config.NEO4J_DATABASE) as session:
            rows = session.run('MATCH (c:ADCRCase {project:$project, dataset_id:$dataset}) '
                'OPTIONAL MATCH (c)-[:HAS_PATH]->(p:ADCRCausalPath) '
                'RETURN c.case_id AS case_id, c.data_json AS data_json, collect(p.payload) AS paths',
                project=config.GRAPH_PROJECT, dataset=dataset)
            for row in rows:
                data = json.loads(row['data_json'])
                data['causal_paths'] = [json.loads(p) for p in row['paths']]
                data = validate_case(data, data.get('source_text'))
                candidates[row['case_id']] = data
    if set(candidates) != set(extracted):
        raise RuntimeError('图谱缺失或数据版本不一致，请先运行 python -m adcr.graph.build_graph')
    # Detect partial rebuilds or graph edits instead of silently comparing different datasets.
    for cid in candidates:
        a = dict(candidates[cid], causal_paths=sorted(candidates[cid]['causal_paths'], key=lambda p:p['path_id']))
        b = dict(extracted[cid], causal_paths=sorted(extracted[cid]['causal_paths'], key=lambda p:p['path_id']))
        if fingerprint(a) != fingerprint(b):
            raise RuntimeError('图谱内容与抽取快照不一致: ' + cid)
    return _rank(case, candidates, top_k, laws, paths_per_case)


def retrieval_key(case, cases, laws, backend, top_k=3):
    query = {k: case.get(k) for k in ('id','facts','question','case_type','source_case_id','exclude_case_ids','event_group')}
    return fingerprint([RETRIEVAL_VERSION, query, dataset_key(cases, laws), backend,
                        top_k, config.PATHS_PER_CASE, retrieval_settings()])


def retrieve(case, top_k=3, backend='auto', extracted=None, laws=None):
    if not isinstance(case, dict):
        raise TypeError('检索需要完整案件对象（facts/question/case_type），不能只传案由')
    if top_k < 1:
        raise ValueError('top_k 必须为正整数')
    retrieval_settings()
    cases = load_cases() if extracted is None else extracted
    laws = load_laws() if laws is None else laws
    if not cases:
        raise RuntimeError('没有可检索案件，请先运行 python -m adcr.ingestion.extract_kg')
    warning = None
    if backend in {'auto', 'neo4j'}:
        try:
            result = retrieve_neo4j(case, cases, top_k, laws, config.PATHS_PER_CASE)
            used = 'neo4j'
        except RetrievalScoringError:
            raise
        except Exception as exc:
            if backend == 'neo4j':
                raise
            warning = str(exc)
            print('WARNING: Neo4j 不可用，实验后端切换为 memory:', warning)
            result = retrieve_memory(case, cases, top_k, laws, config.PATHS_PER_CASE)
            used = 'memory'
    elif backend == 'memory':
        result = retrieve_memory(case, cases, top_k, laws, config.PATHS_PER_CASE)
        used = 'memory'
    else:
        raise ValueError('未知后端: ' + backend)
    result.update(backend=used, fallback_reason=warning, schema_version=2,
                  dataset_id=dataset_key(cases,laws), retrieval_version=RETRIEVAL_VERSION,
                  _fingerprint=retrieval_key(case,cases,laws,backend,top_k))
    return result


def main(backend='neo4j', force=False):
    testset = read_json(config.TESTSET_PATH)
    cases, laws = load_cases(), load_laws()
    dest = config.OUTPUT_DIR / 'retrieval_cache.json'
    cache = read_json(dest, {})
    wanted = {c['id'] for c in testset['cases']}
    cache = {k:v for k,v in cache.items() if k in wanted}
    for case in testset['cases']:
        cid = case['id']
        key = retrieval_key(case,cases,laws,backend,config.TOP_K)
        if force or cache.get(cid,{}).get('_fingerprint') != key or (backend=='auto' and cache.get(cid,{}).get('backend')=='memory'):
            cache[cid] = retrieve(case,config.TOP_K,backend,cases,laws)
            write_json(dest,cache)
        print('RETRIEVE',cid,cache[cid]['backend'],'案件',len(cache[cid]['ranked_cases']),'路径',len(cache[cid]['paths']))
    return cache


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['auto','neo4j','memory'],default='neo4j')
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    main(args.backend,args.force)
