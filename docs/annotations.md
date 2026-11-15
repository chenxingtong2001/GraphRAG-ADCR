# 标注与法规资料

## 案件标注

文书抽取后执行 `python -m adcr.graph.annotations export`，将草稿导出至 `annotations/cases/`。同名人工标注会用于后续建图。

- `node_ids`、`law_entity_ids` 引用实体局部 ID。
- `evidence_quotes` 必须是文书原文中的连续片段。
- 正文变化后，需要重新核对标注与来源的一致性。
- `_annotation_status` 取 `draft` 或 `reviewed`，审校人和时间分别记录在 `reviewed_by`、`reviewed_at`。

测试案可用 `source_case_id`、`exclude_case_ids` 或 `event_group` 排除自身及关联裁判。

## 法规资料

在项目根目录创建 `annotations/`，将 `examples/laws.json` 复制为 `annotations/laws.json`，在 `articles` 中填写法规对象。每项包含 `id`、`name`、`text`、`version`、`source_url` 和 `_annotation_status`。

`version` 记录法规版本及适用时间，`source_url` 指向条文来源。不使用法规资料时，`articles` 保持空数组。
