"""Human annotation overlays are source inputs and survive output cleanup."""
import argparse

from adcr import config
from adcr.graph.schema import validate_case, validate_laws
from adcr.common.storage import fingerprint, read_json, write_json


def load_laws():
    return validate_laws(read_json(config.LAWS_PATH, {"articles": []}))


def load_cases():
    cases = {}
    for fp in sorted(config.EXTRACTED_DIR.glob("*.json")):
        data = read_json(fp)
        source_path = config.PARSED_DIR / (fp.stem + ".txt")
        if not source_path.exists():
            raise ValueError("抽取缺少对应正文: " + fp.name)
        source = source_path.read_text(encoding="utf-8")
        if data.get("_source_hash") != fingerprint(source):
            raise ValueError("正文已变化，请重新抽取: " + fp.name)
        overlay = read_json(config.ANNOTATIONS_DIR / fp.name)
        if overlay:
            if overlay.get("_source_hash") != fingerprint(source):
                raise ValueError("人工标注对应旧正文，需要重新审校: " + fp.name)
            data = overlay
        data = validate_case(data, source)
        if data["case_id"] != fp.stem:
            raise ValueError("case_id 与文件名不一致: " + fp.name)
        cases[data["case_id"]] = data
    return cases


def main(command):
    if command == "export":
        count = 0
        for fp in sorted(config.EXTRACTED_DIR.glob("*.json")):
            dest = config.ANNOTATIONS_DIR / fp.name
            if not dest.exists():
                write_json(dest, read_json(fp))
                count += 1
        print("导出标注文件", count, "个，目录：", config.ANNOTATIONS_DIR)
    else:
        cases = load_cases()
        laws = load_laws()
        print("有效案件", len(cases), "已审校", sum(d["_annotation_status"] == "reviewed" for d in cases.values()))
        print("法规资料", len(laws), "已审校", sum(d.get("_annotation_status") == "reviewed" for d in laws))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["export", "validate"])
    main(parser.parse_args().command)
