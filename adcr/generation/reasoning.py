# -*- coding: utf-8 -*-
"""GraphRAG CLI; LangChain graph retrieval + evidence-grounded generation."""
import argparse
from adcr.generation.generate import SYSTEM_PROMPT, build_prompt, format_paths, main as generate


def main(backend='neo4j', force=False):
    return generate('graphrag',backend,force)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['auto','neo4j','memory'],default='neo4j')
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    main(args.backend,args.force)
