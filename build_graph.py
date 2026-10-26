import json
from neo4j import GraphDatabase
import config
from schema import RELATIONS, normalize_law, validate_case, validate_laws
from storage import fingerprint, read_json

def dataset_key(cases, laws):
    return fingerprint({'schema': 2, 'cases': cases, 'laws': laws})

def get_driver():
    if not config.NEO4J_PASSWORD:
        raise RuntimeError('缺少 NEO4J_PASSWORD')
    return GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD), connection_timeout=10)

def init_constraints(session):
    for (label, key) in [('ADCRCase', 'case_id'), ('ADCREntity', 'entity_id'), ('ADCRCausalPath', 'path_id'), ('ADCRLaw', 'law_id'), ('ADCRConcept', 'concept_id')]:
        session.run('CREATE CONSTRAINT adcr_%s IF NOT EXISTS FOR (n:%s) REQUIRE n.%s IS UNIQUE' % (label.lower(), label, key)).consume()

def build_case(tx, raw, laws=None, dataset_id=''):
    data = validate_case(raw, raw.get('source_text'))
    laws = laws or []
    cid = data['case_id']
    common = dict(case_id=cid, project=config.GRAPH_PROJECT)
    tx.run('MATCH (c:ADCRCase {case_id:$case_id, project:$project})-[r:LAW_RESOURCE]->() DELETE r', **common).consume()
    for label in ('ADCRCausalPath', 'ADCREntity'):
        tx.run('MATCH (n:%s {case_id:$case_id, project:$project}) DETACH DELETE n' % label, **common).consume()
    base = {k: v for (k, v) in data.items() if k != 'causal_paths'}
    tx.run('MERGE (c:ADCRCase {case_id:$case_id}) SET c.project=$project, c.dataset_id=$dataset, c.case_name=$name, c.case_type=$type, c.data_json=$payload', **common, dataset=dataset_id, name=data['case_name'], type=data['case_type'], payload=json.dumps(base, ensure_ascii=False)).consume()
    entities = []
    for e in data['entities']:
        entities.append(dict(entity_id=e['entity_id'], local_id=e['id'], name=e['name'], type=e['type'], desc=e.get('desc', ''), role=e.get('role')))
    tx.run('MATCH (c:ADCRCase {case_id:$case_id}) UNWIND $entities AS item CREATE (e:ADCREntity) SET e=item, e.case_id=$case_id, e.project=$project CREATE (c)-[:HAS_ENTITY {role:item.role}]->(e)', **common, entities=entities).consume()
    for e in data['entities']:
        if e['type'] == 'LawArticle':
            version = 'as_cited_' + str(data.get('year', 'unknown'))
            lid = fingerprint([normalize_law(e['name']), version])
            tx.run('MATCH (e:ADCREntity {entity_id:$eid}) MERGE (l:ADCRLaw {law_id:$lid}) SET l.name=$name, l.version=$version, l.origin="document_citation" MERGE (e)-[:REFERS_TO]->(l)', eid=e['entity_id'], lid=lid, name=e['name'], version=version).consume()
        elif e['type'] in {'Behavior', 'Outcome', 'CausationType'}:
            concept = fingerprint([e['type'], e['name']])
            tx.run('MATCH (e:ADCREntity {entity_id:$eid}) MERGE (n:ADCRConcept {concept_id:$id}) SET n.name=$name, n.type=$type MERGE (e)-[:INSTANCE_OF]->(n)', eid=e['entity_id'], id=concept, name=e['name'], type=e['type']).consume()
    id_map = {e['id']: e['entity_id'] for e in data['entities']}
    for rel in sorted(RELATIONS):
        rows = [dict(src=id_map[r['src']], dst=id_map[r['dst']]) for r in data['relations'] if r['rel'] == rel]
        if rows:
            tx.run('UNWIND $rows AS item MATCH (a:ADCREntity {entity_id:item.src}), (b:ADCREntity {entity_id:item.dst}) MERGE (a)-[r:%s]->(b) SET r.case_id=$case_id, r.source_file=$source' % rel, rows=rows, case_id=cid, source=data.get('_source_file', '')).consume()
    for path in data['causal_paths']:
        tx.run('MATCH (c:ADCRCase {case_id:$case_id}) CREATE (p:ADCRCausalPath {path_id:$pid, case_id:$case_id, project:$project, payload:$payload}) CREATE (c)-[:HAS_PATH]->(p)', **common, pid=path['path_id'], payload=json.dumps(path, ensure_ascii=False)).consume()
        steps = [dict(eid=id_map[n], position=i) for (i, n) in enumerate(path['node_ids'])]
        tx.run('MATCH (p:ADCRCausalPath {path_id:$pid}) UNWIND $steps AS item MATCH (e:ADCREntity {entity_id:item.eid}) CREATE (p)-[:STEP {position:item.position}]->(e)', pid=path['path_id'], steps=steps).consume()
    cited = {normalize_law(e['name']) for e in data['entities'] if e['type'] == 'LawArticle'}
    for law in laws:
        tx.run('MERGE (l:ADCRLaw {law_id:$id}) SET l += $props', id='resource:' + law['id'], props={k: law.get(k, '') for k in ('name', 'text', 'version', 'source_url', '_annotation_status')}).consume()
        if normalize_law(law['name']) in cited:
            tx.run('MATCH (c:ADCRCase {case_id:$cid}), (l:ADCRLaw {law_id:$id}) MERGE (c)-[:LAW_RESOURCE]->(l)', cid=cid, id='resource:' + law['id']).consume()

def main():
    cases = load_cases()
    laws = load_laws()
    if not cases:
        raise RuntimeError('没有抽取结果，请先运行 extract_kg.py')
    dataset = dataset_key(cases, laws)
    with get_driver() as driver:
        driver.verify_connectivity()
        with driver.session(database=config.NEO4J_DATABASE) as session:
            init_constraints(session)
            for data in cases.values():
                session.execute_write(build_case, data, laws, dataset)
                print('GRAPH OK', data['case_id'])
    return dataset

def load_laws():
    return validate_laws(read_json(config.LAWS_PATH, {'articles': []}))

def load_cases():
    cases = {}
    for fp in sorted(config.EXTRACTED_DIR.glob('*.json')):
        data = read_json(fp)
        source_path = config.PARSED_DIR / (fp.stem + '.txt')
        if not source_path.exists():
            raise ValueError('抽取缺少对应正文: ' + fp.name)
        source = source_path.read_text(encoding='utf-8')
        if data.get('_source_hash') != fingerprint(source):
            raise ValueError('正文已变化，请重新抽取: ' + fp.name)
        overlay = read_json(config.ANNOTATIONS_DIR / fp.name)
        if overlay:
            if overlay.get('_source_hash') != fingerprint(source):
                raise ValueError('人工标注对应旧正文，需要重新审校: ' + fp.name)
            data = overlay
        data = validate_case(data, source)
        if data['case_id'] != fp.stem:
            raise ValueError('case_id 与文件名不一致: ' + fp.name)
        cases[data['case_id']] = data
    return cases
