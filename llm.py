import json
import logging
import re
import time
import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
import config
logger = logging.getLogger(__name__)

def get_client():
    config.ensure_llm_configured()
    return OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL, timeout=config.LLM_TIMEOUT, max_retries=0, http_client=httpx.Client(trust_env=config.LLM_TRUST_ENV, follow_redirects=True))

def get_zhipu_client():
    if not config.ZHIPU_API_KEY:
        raise RuntimeError('未配置 ZHIPU_API_KEY；生成可运行，LLM 评估需在 .env 中配置评判模型。')
    return OpenAI(api_key=config.ZHIPU_API_KEY, base_url=config.ZHIPU_BASE_URL, timeout=config.LLM_TIMEOUT, max_retries=0, http_client=httpx.Client(trust_env=config.LLM_TRUST_ENV, follow_redirects=True))

def format_error(exc):
    if isinstance(exc, APITimeoutError):
        hint = '请求超时：检查网络及 LLM_TIMEOUT'
    elif isinstance(exc, APIConnectionError):
        hint = '网络连接失败：检查代理/TLS；当前 LLM_TRUST_ENV=%s' % str(config.LLM_TRUST_ENV).lower()
    elif isinstance(exc, APIStatusError):
        code = exc.status_code
        hint = {401: '鉴权失败：检查对应 API Key', 402: '余额不足：检查服务商账户', 403: '访问被拒绝：检查账户及模型权限', 404: '接口或模型不存在：检查 BASE_URL/MODEL', 429: '请求限流或配额限制：检查服务商返回信息', 400: '请求格式错误：检查模型支持的参数', 422: '请求参数无效：检查模型支持的参数'}.get(code, '服务端 HTTP 错误')
        hint = 'HTTP %s，%s' % (code, hint)
    elif isinstance(exc, ValueError):
        hint = '模型输出/数据校验失败'
    else:
        hint = type(exc).__name__
    messages = []
    seen = set()
    current = exc
    while current is not None and id(current) not in seen and (len(messages) < 5):
        seen.add(id(current))
        message = str(current)
        for secret in (config.DEEPSEEK_API_KEY, config.ZHIPU_API_KEY, config.NEO4J_PASSWORD):
            if secret:
                message = message.replace(secret, '[REDACTED]')
        message = re.sub('(?i)\\bBearer\\s+\\S+', 'Bearer [REDACTED]', message)
        message = re.sub('\\bsk-[A-Za-z0-9_-]+', '[REDACTED]', message)
        if message and message not in messages:
            messages.append(message[:500])
        current = current.__cause__
    return hint + '；' + ' <- '.join(messages)

def _retryable(exc):
    return isinstance(exc, (APIConnectionError, APITimeoutError)) or (isinstance(exc, APIStatusError) and (exc.status_code in {408, 409, 429} or exc.status_code >= 500))

def _request(messages, temperature, max_tokens, client, model, disable_thinking):
    kwargs = dict(model=model or config.DEEPSEEK_MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens)
    if disable_thinking:
        kwargs['extra_body'] = {'thinking': {'type': 'disabled'}}
    started = time.monotonic()
    response = client.chat.completions.create(**kwargs)
    if not response.choices:
        raise ValueError('模型返回空 choices')
    choice = response.choices[0]
    if choice.finish_reason != 'stop':
        raise ValueError('模型输出未正常结束: %s；请调整输出长度或输入' % choice.finish_reason)
    content = choice.message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError('模型返回空内容')
    logger.info('model=%s elapsed=%.2fs usage=%s', model or config.DEEPSEEK_MODEL, time.monotonic() - started, response.usage)
    return content

def _execute(messages, temperature, max_tokens, client, model, disable_thinking, validator=None):
    client = client or get_client()
    if disable_thinking is None:
        disable_thinking = config.LLM_DISABLE_THINKING
    attempts = max(1, config.LLM_MAX_RETRIES)
    messages = list(messages)
    for attempt in range(attempts):
        try:
            raw = _request(messages, temperature, max_tokens, client, model, disable_thinking)
            return validator(raw) if validator else raw
        except Exception as exc:
            if attempt + 1 == attempts or not (_retryable(exc) or isinstance(exc, ValueError)):
                raise
            logger.warning('调用/格式验证失败 %d/%d: %s', attempt + 1, attempts, format_error(exc))
            if validator and isinstance(exc, ValueError):
                messages.append({'role': 'user', 'content': '上次输出未通过格式或证据校验：%s。请重新输出完整 JSON，修正此问题。' % str(exc)[:500]})
            time.sleep(min(2 ** attempt, 8))

def chat(messages, temperature=0.0, max_tokens=4096, client=None, model=None, disable_thinking=None):
    return _execute(messages, temperature, max_tokens, client, model, disable_thinking)

def extract_json(messages, client=None, model=None, disable_thinking=None, validator=None):

    def parse(raw):
        data = _parse_json(raw)
        return validator(data) if validator else data
    return _execute(messages, 0.0, 8192, client, model, disable_thinking, parse)

def _parse_json(raw):
    raw = raw.strip()
    if raw.startswith('```'):
        raw = re.sub('^```[a-zA-Z]*\\s*', '', raw)
        raw = re.sub('\\s*```$', '', raw)
    decoder = json.JSONDecoder()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find('{')
        if start < 0:
            raise ValueError('响应不含 JSON 对象')
        (data, end) = decoder.raw_decode(raw[start:])
        if raw[start + end:].strip():
            raise ValueError('JSON 对象后有额外内容')
    if not isinstance(data, dict):
        raise ValueError('结构化响应必须是 JSON 对象')
    return data
