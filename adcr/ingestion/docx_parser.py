# -*- coding: utf-8 -*-
"""
解析 .docx 裁判文书正文，输出到 data/parsed/*.txt

提取段落与表格内容，保留完整正文（案件事实、裁判依据、裁判说理、裁判结果）。
"""
import re
import sys

from docx import Document
from docx.oxml.ns import qn

from adcr import config
from adcr.common.storage import write_json

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def parse_docx(path):
    """提取 .docx 的全部文本（段落 + 表格），返回字符串。"""
    doc = Document(str(path))
    blocks = []

    # 按文档顺序遍历 body 元素，兼容段落与表格交错
    body = doc.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            text = "".join(node.text or "" for node in child.iter(qn("w:t")))
            if text.strip():
                blocks.append(text)
        elif child.tag == qn("w:tbl"):
            # 表格：逐单元格提取
            for row in child.iter(qn("w:tr")):
                cells = []
                for tc in row.iter(qn("w:tc")):
                    cell_text = "".join(node.text or "" for node in tc.iter(qn("w:t")))
                    cells.append(cell_text.strip())
                if cells:
                    blocks.append("\t".join(cells))
    return "\n".join(blocks)


def clean_text(text):
    """基础清洗：压缩多余空行，统一换行。"""
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main():
    files = sorted(config.DOCX_DIR.glob("*.docx"))
    if not files:
        raise RuntimeError("未在目录 %s 找到 .docx 文件" % config.DOCX_DIR)

    config.PARSED_DIR.mkdir(parents=True, exist_ok=True)
    parsed_meta = {}
    ok = 0
    for fp in files:
        try:
            text = parse_docx(fp)
            text = clean_text(text)
            if not text:
                print("空文本（可能为扫描件）:", fp.name)
                continue
            out_path = config.PARSED_DIR / (fp.stem + ".txt")
            out_path.write_text(text, encoding="utf-8")
            parsed_meta[fp.stem] = {"file": fp.name, "chars": len(text)}
            ok += 1
            print("OK  %-60s %d chars" % (fp.name, len(text)))
        except Exception as e:  # noqa: BLE001
            print("FAIL %s -> %s" % (fp.name, e))

    meta_path = config.PARSED_DIR / "_meta.json"
    write_json(meta_path, parsed_meta)
    print("\n解析完成：成功 %d / 共 %d，正文已写入 %s" % (ok, len(files), config.PARSED_DIR))
    if ok != len(files):
        raise RuntimeError("部分文书解析失败或为空，请检查后重跑")


if __name__ == "__main__":
    main()
