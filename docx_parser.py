import re
import sys
from docx import Document
from docx.oxml.ns import qn
import config
from storage import write_json
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

def parse_docx(path):
    doc = Document(str(path))
    blocks = []
    body = doc.element.body
    for child in body.iterchildren():
        if child.tag == qn('w:p'):
            text = ''.join((node.text or '' for node in child.iter(qn('w:t'))))
            if text.strip():
                blocks.append(text)
        elif child.tag == qn('w:tbl'):
            for row in child.iter(qn('w:tr')):
                cells = []
                for tc in row.iter(qn('w:tc')):
                    cell_text = ''.join((node.text or '' for node in tc.iter(qn('w:t'))))
                    cells.append(cell_text.strip())
                if cells:
                    blocks.append('\t'.join(cells))
    return '\n'.join(blocks)

def clean_text(text):
    text = text.replace('\u3000', ' ')
    text = re.sub('[ \\t]+', ' ', text)
    text = re.sub('\\n{3,}', '\n\n', text)
    return text.strip()

def main():
    files = sorted(config.DOCX_DIR.glob('*.docx'))
    if not files:
        raise RuntimeError('未在目录 %s 找到 .docx 文件' % config.DOCX_DIR)
    config.PARSED_DIR.mkdir(parents=True, exist_ok=True)
    parsed_meta = {}
    ok = 0
    for fp in files:
        try:
            text = parse_docx(fp)
            text = clean_text(text)
            if not text:
                print('空文本（可能为扫描件）:', fp.name)
                continue
            out_path = config.PARSED_DIR / (fp.stem + '.txt')
            out_path.write_text(text, encoding='utf-8')
            parsed_meta[fp.stem] = {'file': fp.name, 'chars': len(text)}
            ok += 1
            print('OK  %-60s %d chars' % (fp.name, len(text)))
        except Exception as e:
            print('FAIL %s -> %s' % (fp.name, e))
    meta_path = config.PARSED_DIR / '_meta.json'
    write_json(meta_path, parsed_meta)
    print('\n解析完成：成功 %d / 共 %d，正文已写入 %s' % (ok, len(files), config.PARSED_DIR))
    if ok != len(files):
        raise RuntimeError('部分文书解析失败或为空，请检查后重跑')
