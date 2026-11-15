# -*- coding: utf-8 -*-
"""Pure LLM control, sharing model, task instructions and output schema with GraphRAG."""
import argparse
from adcr.generation.generate import SYSTEM_PROMPT, build_prompt, main as generate


def main(force=False):
    return generate('baseline',force=force)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--force',action='store_true')
    main(parser.parse_args().force)
