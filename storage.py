import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

def fingerprint(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()

def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    (fd, temporary) = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def utc_now():
    return datetime.now(timezone.utc).isoformat()
