# -*- coding: utf-8 -*-
"""模型、数据库及检索参数。"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 项目路径
BASE_DIR = Path(__file__).resolve().parents[1]

# 环境变量
load_dotenv(BASE_DIR / ".env")


def _env(key, default=None):
    val = os.environ.get(key)
    return val if val not in (None, "") else default


# 生成接口
DEEPSEEK_API_KEY = _env("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = _env("DEEPSEEK_MODEL", "your-generation-model")
LLM_TIMEOUT = int(_env("LLM_TIMEOUT", "120"))
LLM_MAX_RETRIES = int(_env("LLM_MAX_RETRIES", "3"))
LLM_DISABLE_THINKING = _env("LLM_DISABLE_THINKING", "true").lower() == "true"
LLM_TRUST_ENV = _env("LLM_TRUST_ENV", "false").lower() == "true"

# 评判接口
ZHIPU_API_KEY = _env("ZHIPU_API_KEY", "")
ZHIPU_BASE_URL = _env("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
ZHIPU_MODEL = _env("ZHIPU_MODEL", "your-judge-model")

# Neo4j
NEO4J_URI = _env("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = _env("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = _env("NEO4J_PASSWORD", "")
NEO4J_DATABASE = _env("NEO4J_DATABASE", "neo4j")

# 数据路径
# DOCX 文书
DOCX_DIR = BASE_DIR / "corpus"
# 中间数据
DATA_DIR = BASE_DIR / "data"
PARSED_DIR = DATA_DIR / "parsed"          # 解析后的正文文本
EXTRACTED_DIR = DATA_DIR / "extracted"    # LLM 抽取的结构化 JSON
# 输出
OUTPUT_DIR = BASE_DIR / "output"
BASELINE_RESULTS = OUTPUT_DIR / "baseline_results.json"
GRAPHRAG_RESULTS = OUTPUT_DIR / "graphrag_results.json"
EVAL_REPORT = OUTPUT_DIR / "evaluation_report.md"
COMPARISON_CSV = OUTPUT_DIR / "comparison.csv"
# 测试集
TESTSET_PATH = BASE_DIR / "testset.json"

ANNOTATIONS_DIR = BASE_DIR / "annotations" / "cases"
LAWS_PATH = BASE_DIR / "annotations" / "laws.json"
GRAPH_PROJECT = "graphrag-adcr-v2"
TOP_K = 3
PATHS_PER_CASE = 2

# 混合检索
EMBEDDING_MODEL = _env('EMBEDDING_MODEL', 'BAAI/bge-small-zh-v1.5')
EMBEDDING_REVISION = _env('EMBEDDING_REVISION', 'main')
EMBEDDING_DEVICE = _env('EMBEDDING_DEVICE', 'cpu')
EMBEDDING_BATCH_SIZE = int(_env('EMBEDDING_BATCH_SIZE', '16'))
EMBEDDING_CHUNK_TOKENS = int(_env('EMBEDDING_CHUNK_TOKENS', '384'))
EMBEDDING_CHUNK_OVERLAP = int(_env('EMBEDDING_CHUNK_OVERLAP', '64'))
EMBEDDING_QUERY_PREFIX = os.environ.get('EMBEDDING_QUERY_PREFIX', '为这个句子生成表示以用于检索相关文章：')
HYBRID_KEYWORD_WEIGHT = float(_env('HYBRID_KEYWORD_WEIGHT', '0.5'))
HYBRID_SEMANTIC_WEIGHT = float(_env('HYBRID_SEMANTIC_WEIGHT', '0.5'))
HYBRID_RRF_K = int(_env('HYBRID_RRF_K', '60'))
HYBRID_CANDIDATES = int(_env('HYBRID_CANDIDATES', '10'))


def ensure_llm_configured():
    """检查生成接口凭据。"""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "未检测到 DEEPSEEK_API_KEY。请复制 .env.example 为 .env，并填写 DEEPSEEK_API_KEY 后重试。"
        )
