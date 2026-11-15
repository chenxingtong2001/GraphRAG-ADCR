# -*- coding: utf-8 -*-
"""Run the complete experiment: python run.py."""
import logging

from adcr.generation import baseline
from adcr.graph import build_graph
from adcr import config
from adcr.ingestion import docx_parser
from adcr.evaluation import evaluate
from adcr.ingestion import extract_kg
from adcr.generation import reasoning


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    steps = [
        ('解析文书', docx_parser.main),
        ('抽取知识', extract_kg.main),
        ('构建图谱', build_graph.main),
        ('生成基线答案', baseline.main),
        ('混合检索类案（BM25 + 语义向量）并生成 GraphRAG 答案', lambda: reasoning.main(backend='neo4j')),
        ('评估并生成报告', evaluate.main),
    ]
    for index, (name, execute) in enumerate(steps, 1):
        print('\n[%d/%d] %s' % (index, len(steps), name), flush=True)
        execute()
    print('\n运行完成，结果目录：%s' % config.OUTPUT_DIR)


if __name__ == '__main__':
    main()
