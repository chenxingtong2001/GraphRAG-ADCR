"""Local dense retrieval with token windows and content-addressed embedding cache."""
import math
from functools import lru_cache

from adcr import config
from adcr.common.storage import fingerprint, read_json, write_json

EMBEDDING_VERSION = 'token-windows-query-mean-doc-max'


def settings():
    return dict(version=EMBEDDING_VERSION, model=config.EMBEDDING_MODEL,
                revision=config.EMBEDDING_REVISION, device=config.EMBEDDING_DEVICE,
                chunk_tokens=config.EMBEDDING_CHUNK_TOKENS,
                overlap=config.EMBEDDING_CHUNK_OVERLAP,
                query_prefix=config.EMBEDDING_QUERY_PREFIX)


@lru_cache(maxsize=2)
def _load_model(name, revision, device):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError('语义检索依赖未安装，请执行 python -m pip install -r requirements.txt') from exc
    try:
        return SentenceTransformer(name, revision=revision, device=device, trust_remote_code=False,
                                   cache_folder=str(config.BASE_DIR / '.cache' / 'models'))
    except Exception as exc:
        raise RuntimeError('无法加载语义模型 %s；请检查模型下载连接或将 EMBEDDING_MODEL 设为本地模型目录' % name) from exc


def _validate_vectors(vectors, expected=None):
    if not isinstance(vectors, list) or not vectors:
        raise ValueError('语义向量不能为空')
    dim = expected or len(vectors[0])
    for vector in vectors:
        if not isinstance(vector, list) or len(vector) != dim or not dim:
            raise ValueError('语义向量维度不一致')
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vector):
            raise ValueError('语义向量包含无效数值')
        norm = math.sqrt(sum(v*v for v in vector))
        if not math.isclose(norm, 1, abs_tol=1e-4):
            raise ValueError('语义向量必须归一化且非零')
    return dim


def encode_texts(texts, query=False):
    """Return a list of chunk vectors per text; no long-document tail is discarded."""
    if not texts:
        return []
    options = settings()
    model = _load_model(options['model'], options['revision'], options['device'])
    prefix = options['query_prefix'] if query else ''
    tokenizer = model.tokenizer
    prefix_size = len(tokenizer.encode(prefix, add_special_tokens=False))
    limit = int(model.max_seq_length) - tokenizer.num_special_tokens_to_add(pair=False)
    window = min(options['chunk_tokens'], limit - prefix_size)
    overlap = options['overlap']
    if window < 1 or overlap < 0 or overlap >= window or config.EMBEDDING_BATCH_SIZE < 1:
        raise ValueError('语义分块需要 0 <= overlap < window，且 batch_size 为正数')
    path = config.DATA_DIR / 'embeddings' / (fingerprint(options) + '.json')
    cache = read_json(path, {})
    keys = [fingerprint([text, query]) for text in texts]
    pending, spans = [], {}
    for key, text in zip(keys, texts):
        if key in cache or key in spans:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError('语义检索文本不能为空')
        start = len(pending)
        for offset in range(0, len(ids), window-overlap):
            chunk = prefix + tokenizer.decode(ids[offset:offset+window], skip_special_tokens=True)
            # Decode/re-tokenize may change token count; reject rather than silently truncate.
            if len(tokenizer.encode(chunk, add_special_tokens=False)) > limit:
                raise ValueError('分块重新分词后超出模型长度，请调小 EMBEDDING_CHUNK_TOKENS')
            pending.append(chunk)
            if offset + window >= len(ids):
                break
        spans[key] = (start, len(pending))
    if pending:
        vectors = model.encode(pending, batch_size=config.EMBEDDING_BATCH_SIZE,
                               normalize_embeddings=True, show_progress_bar=False).tolist()
        if len(vectors) != len(pending):
            raise ValueError('语义模型返回的向量数量不一致')
        _validate_vectors(vectors)
        for key, (start, end) in spans.items():
            cache[key] = vectors[start:end]
    result = [cache[key] for key in keys]
    dim = None
    for vectors in result:
        dim = _validate_vectors(vectors, dim)
    if pending:
        write_json(path, cache)
    return result


def semantic_scores(query, documents):
    if not documents:
        return []
    queries = encode_texts([query], query=True)[0]
    docs = encode_texts(documents)
    for chunks in docs:
        _validate_vectors(chunks, len(queries[0]))
    # Each query window finds its best document window; average over query windows.
    return [sum(max(sum(a*b for a,b in zip(q,d)) for d in chunks)
                for q in queries) / len(queries) for chunks in docs]
