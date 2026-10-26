import copy
import re
from storage import fingerprint
SCHEMA_VERSION = 2
ENTITY_TYPES = {'Party', 'Behavior', 'Outcome', 'LawArticle', 'CausationType'}
RELATIONS = {'INVOLVES', 'COMMITS', 'CAUSES', 'HAS_CAUSATION', 'CITES'}
PATTERNS = {'single_single', 'multiple_multiple', 'single_multiple', 'multiple_single'}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def text(value, label):
    require(isinstance(value, str) and bool(value.strip()), label + ' 必须为非空字符串')
    return value.strip()

def validate_case(raw, source_text=None):
    require(isinstance(raw, dict), '抽取结果必须是对象')
    data = copy.deepcopy(raw)
    cid = text(data.get('case_id'), 'case_id')
    for key in ('case_name', 'case_type'):
        text(data.get(key), key)
    require(data.get('schema_version') == SCHEMA_VERSION, '抽取格式过旧，请重新运行 extract_kg.py')
    for key in ('entities', 'relations', 'causal_paths'):
        require(isinstance(data.get(key), list), key + ' 必须为列表')
    require(bool(data['entities']), 'entities 不能为空')
    require(bool(data['causal_paths']), 'causal_paths 不能为空')
    ids = {}
    for entity in data['entities']:
        require(isinstance(entity, dict), '实体必须为对象')
        eid = text(entity.get('id'), 'entity.id')
        require(eid not in ids, '重复实体 id: ' + eid)
        require(entity.get('type') in ENTITY_TYPES, '非法实体类型')
        text(entity.get('name'), 'entity.name')
        require(entity.get('role') in {None, 'plaintiff', 'defendant', 'other'}, '非法角色（其他用 other，无角色用 null）')
        require(isinstance(entity.get('desc', ''), str), 'entity.desc 必须为字符串')
        entity['entity_id'] = fingerprint([cid, eid])
        ids[eid] = entity
    for relation in data['relations']:
        require(isinstance(relation, dict), '关系必须为对象')
        require(relation.get('rel') in RELATIONS, '非法关系类型')
        require(relation.get('src') in ids and relation.get('dst') in ids, '关系引用不存在的实体 id')
    path_ids = set()
    for path in data['causal_paths']:
        require(isinstance(path, dict), '路径必须为对象')
        pid = text(path.get('id'), 'path.id')
        require(pid not in path_ids, '重复路径 id')
        path_ids.add(pid)
        nodes = path.get('node_ids')
        require(isinstance(nodes, list) and len(nodes) >= 2, '路径必须至少包含两个 node_ids')
        require(all((isinstance(n, str) and n in ids for n in nodes)), '路径引用不存在的实体 id')
        require(path.get('causal_type') in {'direct', 'indirect'}, '非法因果类型')
        require(path.get('causal_pattern') in PATTERNS, '非法因果模式')
        text(path.get('explanation'), 'path.explanation')
        laws = path.get('law_entity_ids', [])
        require(isinstance(laws, list), 'law_entity_ids 必须为列表')
        require(all((n in ids and ids[n]['type'] == 'LawArticle' for n in laws)), '路径法条引用无效')
        quotes = path.get('evidence_quotes')
        require(isinstance(quotes, list) and bool(quotes), '每条路径必须提供原文证据 evidence_quotes')
        for quote in quotes:
            text(quote, 'evidence_quote')
            if source_text is not None:
                require(quote in source_text, '路径证据不是原文中的连续片段: ' + quote[:40])
        path['path_id'] = fingerprint([cid, pid])
        path['chain'] = [ids[n]['name'] for n in nodes]
        path['law_articles'] = [ids[n]['name'] for n in laws]
    data.setdefault('_annotation_status', 'draft')
    if source_text is not None:
        data['source_text'] = source_text
    return data

def normalize_law(name):
    return re.sub('[\\s《》〈〉]', '', name).replace('中华人民共和国', '')

def validate_laws(raw):
    require(isinstance(raw, dict) and isinstance(raw.get('articles'), list), 'laws.json 应包含 articles 列表')
    result = []
    ids = set()
    for article in raw['articles']:
        require(isinstance(article, dict), '法条应为对象')
        for key in ('id', 'name', 'text', 'version', 'source_url'):
            text(article.get(key), 'law.' + key)
        require(article['id'] not in ids, '重复法条 id')
        ids.add(article['id'])
        result.append(copy.deepcopy(article))
    return result
