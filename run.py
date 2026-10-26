import logging
import build_graph
import config
import docx_parser
import evaluate
import extract_kg
import generation

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    steps = [('解析文书', docx_parser.main), ('抽取知识', extract_kg.main), ('构建图谱', build_graph.main), ('生成基线答案', lambda : generation.main('baseline')), ('检索并生成 GraphRAG 答案', lambda : generation.main('graphrag')), ('评估', evaluate.main)]
    for (index, (name, execute)) in enumerate(steps, 1):
        print('\n[%d/%d] %s' % (index, len(steps), name), flush=True)
        execute()
    print('结果：', config.OUTPUT_DIR)
if __name__ == '__main__':
    main()
