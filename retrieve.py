import collections
import json
import math
import re
import config
from build_graph import dataset_key, get_driver, load_cases, load_laws
from schema import validate_case
from storage import fingerprint, read_json, write_json
RETRIEVAL_VERSION = 'facts-bm25-path-v2'
STOP = {'责任', '认定', '因果', '关系', '案件', '如何', '应当', '承担', '纠纷', '问题', '本案'}

def tokenize(text):
    result = re.findall('[a-z0-9]+', text.lower())
    for run in re.findall('[\\u4e00-\\u9fff]+', text):
        result.extend((run[i:i + 2] for i in range(len(run) - 1) if run[i:i + 2] not in STOP))
    return result

def _bm25(query, documents):
    counters = [collections.Counter(tokenize(d)) for d in documents]
    lengths = [sum(c.values()) for c in counters]
    avg = sum(lengths) / max(len(lengths), 1) or 1
    terms = set(tokenize(query))
    df = {t: sum((t in c for c in counters)) for t in terms}
    scores = []
    for (counter, length) in zip(counters, lengths):
        score = 0.0
        for term in terms:
            tf = counter[term]
            if tf:
                idf = math.log(1 + (len(counters) - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * length / avg))
        scores.append(score)
    return scores

def _query_text(case):
    return '\n'.join((case.get(k, '') for k in ('facts', 'question', 'case_type')))

def _path_text(path):
    return ' '.join(path['chain'] + path.get('law_articles', []) + [path['explanation']] + path.get('evidence_quotes', []))

def _document(data):
    return ' '.join([data['case_name'], data['case_type'], data.get('source_text', '')] + [e['name'] + ' ' + e.get('desc', '') for e in data['entities']])

def _rank(case, candidates, top_k, laws, paths_per_case):
    query = _query_text(case)
    qt = set(tokenize(query))
    ordered = sorted(candidates.items())
    bm25 = _bm25(query, [_document(d) for (_, d) in ordered])
    excluded = set(case.get('exclude_case_ids', [])) | {case.get('source_case_id', '')}
    scored = []
    for ((cid, data), lexical) in zip(ordered, bm25):
        if cid in excluded or (case.get('event_group') and data.get('event_group') == case['event_group']):
            continue
        cps = []
        for path in data['causal_paths']:
            overlap = qt & set(tokenize(_path_text(path)))
            cps.append((len(overlap) / max(len(qt), 1), path))
        cps.sort(key=lambda item: (-item[0], item[1]['path_id']))
        if not cps or lexical <= 0:
            continue
        type_tokens = set(tokenize(case.get('case_type', '')))
        candidate_tokens = set(tokenize(data['case_type']))
        type_score = len(type_tokens & candidate_tokens) / max(len(type_tokens | candidate_tokens), 1)
        score = 0.7 * lexical / (lexical + 5) + 0.15 * type_score + 0.15 * cps[0][0]
        matched = sorted(qt & set(tokenize(_document(data))))
        scored.append((score, cid, data, cps, matched))
    scored.sort(key=lambda row: (-row[0], row[1]))
    (ranked, paths) = ([], [])
    for (rank, (score, cid, data, cps, matched)) in enumerate(scored[:top_k], 1):
        ranked.append(dict(case_id=cid, case_name=data['case_name'], rank=rank, similarity=round(score, 6), matched_terms=matched[:30], annotation_status=data['_annotation_status']))
        for (ps, path) in cps[:paths_per_case]:
            paths.append(dict(path, case_id=cid, case_name=data['case_name'], rank=rank, similarity=round(score, 6), path_similarity=round(ps, 6), annotation_status=data['_annotation_status'], source_file=data.get('_source_file', '')))
    cited = {name for p in paths for name in p.get('law_articles', [])}
    resources = []
    for law in laws:
        overlap = len(qt & set(tokenize(law['name'] + law['text'])))
        if overlap or law['name'] in cited:
            resources.append((overlap, law))
    resources.sort(key=lambda row: (-row[0], row[1]['id']))
    return dict(ranked_cases=ranked, paths=paths, laws=[law for (_, law) in resources[:3]])

def retrieve_neo4j(case, extracted, top_k=3, laws=None, paths_per_case=2):
    laws = laws or []
    dataset = dataset_key(extracted, laws)
    candidates = {}
    with get_driver() as driver:
        driver.verify_connectivity()
        with driver.session(database=config.NEO4J_DATABASE) as session:
            rows = session.run('MATCH (c:ADCRCase {project:$project, dataset_id:$dataset}) OPTIONAL MATCH (c)-[:HAS_PATH]->(p:ADCRCausalPath) RETURN c.case_id AS case_id, c.data_json AS data_json, collect(p.payload) AS paths', project=config.GRAPH_PROJECT, dataset=dataset)
            for row in rows:
                data = json.loads(row['data_json'])
                data['causal_paths'] = [json.loads(p) for p in row['paths']]
                data = validate_case(data, data.get('source_text'))
                candidates[row['case_id']] = data
    if set(candidates) != set(extracted):
        raise RuntimeError('图谱缺失或数据版本不一致，请先运行 build_graph.py')
    for cid in candidates:
        a = dict(candidates[cid], causal_paths=sorted(candidates[cid]['causal_paths'], key=lambda p: p['path_id']))
        b = dict(extracted[cid], causal_paths=sorted(extracted[cid]['causal_paths'], key=lambda p: p['path_id']))
        if fingerprint(a) != fingerprint(b):
            raise RuntimeError('图谱内容与抽取快照不一致: ' + cid)
    return _rank(case, candidates, top_k, laws, paths_per_case)

def retrieval_key(case, cases, laws):
    query = {key: case.get(key) for key in ('id', 'facts', 'question', 'case_type', 'source_case_id', 'exclude_case_ids', 'event_group')}
    return fingerprint([RETRIEVAL_VERSION, query, dataset_key(cases, laws), config.TOP_K, config.PATHS_PER_CASE])

def main():
    testset = read_json(config.TESTSET_PATH)
    (cases, laws) = (load_cases(), load_laws())
    dest = config.OUTPUT_DIR / 'retrieval_cache.json'
    wanted = {case['id'] for case in testset['cases']}
    cache = {key: value for (key, value) in read_json(dest, {}).items() if key in wanted}
    for case in testset['cases']:
        cid = case['id']
        key = retrieval_key(case, cases, laws)
        if cache.get(cid, {}).get('_fingerprint') != key:
            entry = retrieve_neo4j(case, cases, config.TOP_K, laws, config.PATHS_PER_CASE)
            entry.update(backend='neo4j', schema_version=2, dataset_id=dataset_key(cases, laws), retrieval_version=RETRIEVAL_VERSION, _fingerprint=key)
            cache[cid] = entry
            write_json(dest, cache)
        print('RETRIEVE', cid, '案件', len(cache[cid]['ranked_cases']), '路径', len(cache[cid]['paths']))
    return cache
