<div align="center">

# GraphRAG-ADCR

**自动驾驶事故 · 法律因果推理 · 可追溯类案证据**

从司法文书构建知识图谱，检索类案因果路径，对比 GraphRAG 与纯 LLM 的分析结果。

[快速开始](#快速开始) · [方法流程](#方法流程) · [输入说明](#输入说明) · [结果查看](#结果查看)

</div>

![GraphRAG-ADCR 方法流程](assets/graphrag-adcr.svg)

> 本项目依据 [A GraphRAG Approach to Enhancing the Ability of Causal Reasoning in Autonomous Driving Cases](https://doi.org/10.3233/FAIA251625) 实现方法流程。检索公式和参数为本项目的实现选择；仓库不附带论文原始数据或实验结果。

## 方法流程

核心思路是先找到与当前案件有关的类案及原文证据，再让模型基于这些材料分析因果关系与责任。

| 阶段 | 做什么 | 对应代码 |
| :--- | :--- | :--- |
| ① 解析 | 按顺序读取 DOCX 中的段落与表格 | `docx_parser.py` |
| ② 抽取 | 提取当事人、行为、结果、法条及带原文证据的因果路径 | `extract_kg.py` |
| ③ 建图 | 将案件、实体、关系和因果路径写入 Neo4j | `build_graph.py` |
| ④ 检索 | 根据案件事实、案由与路径重叠检索相关类案 | `retrieve.py` |
| ⑤ 生成 | 同一生成模型分别给出纯 LLM 与 GraphRAG 回答 | `generation.py` |
| ⑥ 评估 | 按可用标注计算指标，生成报告与对比表 | `evaluate.py` |

GraphRAG 回答包含类案匹配分析、因果链、路径引用、责任主体和法律依据。参考答案仅用于评估，不传入生成提示词。

## 快速开始

### 1. 准备环境

需要 **Python 3.9+**、**Neo4j 5+** 及可用的模型 API。以下命令在项目根目录执行，使用 PowerShell：

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
Copy-Item testset.example.json testset.json
New-Item -ItemType Directory -Force corpus
```

### 2. 填写配置

编辑 `.env`，填写以下项目，并启动 Neo4j：

| 配置 | 用途 | 是否必填 |
| :--- | :--- | :--- |
| `DEEPSEEK_API_KEY` | 知识抽取及两组回答生成 | 是 |
| `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL` | 生成接口地址与模型 | 已提供默认值 |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` / `NEO4J_DATABASE` | 数据库连接 | 填写密码，其余按本机配置调整 |
| `ZHIPU_API_KEY` | 独立评判模型 | 否；留空时仅计算离线指标 |
| `ZHIPU_BASE_URL` / `ZHIPU_MODEL` | 评判接口地址与模型 | 使用评判模型时核对 |

### 3. 放入输入

- 将自备的 `.docx` 文书放入 `corpus/`。
- 将 `testset.json` 的虚构示例替换为待分析案件，字段说明见下文。
- 如需评估生成质量，补充经过核对的参考答案。

### 4. 一键运行

```powershell
python run.py
```

终端按六个阶段显示进度，结果保存在 `output/`。发生错误后修正问题，重新执行同一命令即可复用有效缓存、重试失败项。

## 输入说明

### 待分析案件

`testset.example.json` 仅提供虚构的格式示例。`testset.json` 的基本结构如下：

```json
{
  "cases": [
    {
      "id": "example",
      "case_type": "交通事故责任纠纷",
      "facts": "虚构示例：车辆发生碰撞，原因及责任主体尚待查明。",
      "question": "需要哪些证据分析事故因果关系与责任？"
    }
  ]
}
```

每个案件的 `id` 应唯一。按需要增加以下字段：

| 可选字段 | 内容 | 用途 |
| :--- | :--- | :--- |
| `legal_context` | 法律版本、适用时间等前提 | 约束生成与评估的法律语境 |
| `reference_answer` | 完整参考答案文本 | ROUGE、模型评分及匿名定性比较 |
| `reference.responsible_parties` | 责任主体名称数组 | 主体集合 P / R / F1 |
| `reference.law_articles` | 引用法条名称数组 | 法条集合 P / R / F1 |
| `reference.liability_conclusion` | 责任结论文本 | 模型判定结论一致性 |
| `relevant_case_ids` | 相关语料文件名数组，不含 `.docx` | 检索 P@3 / R@3 / MRR |

未填写的标注对应指标记为缺失。模型评分、结论一致性及匿名定性比较还需要配置评判模型；评分维度为完备性、可解释性和准确性。

### 人工修订与法规资源

- **修订抽取结果**：将 `data/extracted/` 中的 JSON 复制到 `annotations/cases/` 后修改，保留文件名及来源指纹 `_source_hash`。
- **补充法规**：可创建 `laws.json`，格式为 `{"articles": []}`；每条法规填写 `id`、`name`、`text`、`version`、`source_url`。

这两项均可省略。模型抽取结果默认记录为 `draft`，人工审校状态由实际审校者维护。

## 结果查看

| 文件 | 内容 |
| :--- | :--- |
| `output/evaluation_report.md` | **优先查看**：整体指标、逐案比较及检索证据 |
| `output/comparison.csv` | 逐案指标表，可用 Excel 打开 |
| `output/baseline_results.json` | 纯 LLM 的回答与结构化结果 |
| `output/graphrag_results.json` | GraphRAG 回答、引用路径及实际检索快照 |
| `output/retrieval_cache.json` | 有序类案匹配、因果路径和原文证据 |
| `output/evaluation_details.json` | 各项评分、调用状态和失败原因 |

`data/parsed/` 保存解析正文，`data/extracted/` 保存抽取结果。两者用于续跑，可保留到实验结束。

<details>
<summary><strong>项目文件与常用调整</strong></summary>

| 文件 | 职责 |
| :--- | :--- |
| `run.py` | 统一运行入口 |
| `config.py` | 环境变量、路径与检索参数 |
| `llm.py` | 模型调用、重试与错误脱敏 |
| `schema.py` | 结构及原文证据校验 |
| `storage.py` | 缓存指纹与原子写入 |
| `metrics.py` | 离线评估指标 |

默认从 Neo4j 检索前 3 个类案，每案最多 2 条因果路径，可在 `config.py` 中调整 `TOP_K`、`PATHS_PER_CASE`。

默认直连模型接口。如需继承系统代理，在 `.env` 中设置 `LLM_TRUST_ENV=true`。接口不支持关闭思考参数时，设置 `LLM_DISABLE_THINKING=false`；需要更长请求时间时，可设置 `LLM_TIMEOUT`。

</details>

## 数据与许可

仓库仅提供代码框架和虚构输入示例。密钥、语料、私人测试集、人工标注及运行产物均已加入 `.gitignore`。

模型评分是辅助评价，不能替代法律专家判断。本版本沿用原项目 [CC BY-NC 4.0](LICENSE) 许可证。
