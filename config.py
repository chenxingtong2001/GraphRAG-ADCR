import os
from pathlib import Path
from dotenv import load_dotenv
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env')

def env(key, default=''):
    return os.getenv(key) or default
DEEPSEEK_API_KEY = env('DEEPSEEK_API_KEY')
DEEPSEEK_BASE_URL = env('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
DEEPSEEK_MODEL = env('DEEPSEEK_MODEL', 'deepseek-v4-pro')
ZHIPU_API_KEY = env('ZHIPU_API_KEY')
ZHIPU_BASE_URL = env('ZHIPU_BASE_URL', 'https://open.bigmodel.cn/api/paas/v4/')
ZHIPU_MODEL = env('ZHIPU_MODEL', 'glm-5.2')
LLM_TIMEOUT = int(env('LLM_TIMEOUT', '120'))
LLM_MAX_RETRIES = int(env('LLM_MAX_RETRIES', '3'))
LLM_DISABLE_THINKING = env('LLM_DISABLE_THINKING', 'true').lower() == 'true'
LLM_TRUST_ENV = env('LLM_TRUST_ENV', 'false').lower() == 'true'
NEO4J_URI = env('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = env('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = env('NEO4J_PASSWORD')
NEO4J_DATABASE = env('NEO4J_DATABASE', 'neo4j')
DOCX_DIR = BASE_DIR / 'corpus'
DATA_DIR = BASE_DIR / 'data'
PARSED_DIR = DATA_DIR / 'parsed'
EXTRACTED_DIR = DATA_DIR / 'extracted'
OUTPUT_DIR = BASE_DIR / 'output'
BASELINE_RESULTS = OUTPUT_DIR / 'baseline_results.json'
GRAPHRAG_RESULTS = OUTPUT_DIR / 'graphrag_results.json'
EVAL_REPORT = OUTPUT_DIR / 'evaluation_report.md'
COMPARISON_CSV = OUTPUT_DIR / 'comparison.csv'
TESTSET_PATH = BASE_DIR / 'testset.json'
ANNOTATIONS_DIR = BASE_DIR / 'annotations' / 'cases'
LAWS_PATH = BASE_DIR / 'laws.json'
GRAPH_PROJECT = 'graphrag-adcr-v2'
TOP_K = 3
PATHS_PER_CASE = 2

def ensure_llm_configured():
    if not DEEPSEEK_API_KEY:
        raise RuntimeError('请在 .env 中填写 DEEPSEEK_API_KEY')
