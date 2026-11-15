"""Offline regression tests: synthetic fixtures only; no paid API or live Neo4j."""
import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from adcr.graph import annotations
from adcr.generation import baseline
from adcr.graph import build_graph
from scripts import clean_outputs
from adcr import config
from adcr.ingestion import docx_parser
from adcr.evaluation import evaluate
from adcr.ingestion import extract_kg
from adcr.generation import generate as generation
from adcr.common import llm
from adcr.generation import reasoning
from adcr.retrieval import retrieve
import run
from adcr.graph.schema import validate_case
from adcr.common.storage import fingerprint, read_json, write_json


def fixture(cid='document', fact='硬件缺陷导致碰撞', reviewed=False):
    raw = dict(schema_version=2, case_id=cid, case_name=cid, case_type='交通事故',
        court='测试法院', year=2020, source_text=fact, _source_file=cid+'.txt',
        _source_hash=fingerprint(fact), _annotation_status='reviewed' if reviewed else 'draft',
        entities=[dict(id='p',name='A公司',type='Party',role='defendant',desc='测试主体'),
                  dict(id='b',name=fact[:-4],type='Behavior',role=None,desc=fact),
                  dict(id='o',name='碰撞',type='Outcome',role=None,desc='测试结果')],
        relations=[dict(src='p',dst='b',rel='COMMITS'),dict(src='b',dst='o',rel='CAUSES')],
        causal_paths=[dict(id='path',node_ids=['b','o'],causal_type='direct',
            causal_pattern='single_single',law_entity_ids=[],explanation=fact,evidence_quotes=[fact])])
    if reviewed:raw.update(reviewed_by='测试审校者',reviewed_at='2026-10-26')
    return validate_case(raw,fact)


def answer(entry=None):
    return dict(case_matching_analysis='根据事实与参考路径分析。',
        case_matching_citations=[],
        reasoning_paths=[dict(chain=['测试行为','测试结果'],causal_type='direct',
            causal_pattern='single_single',explanation='测试因果论证',
            citation_note='类案仅支持一般框架，仍需结合本案事实。' if entry and entry['paths'] else '本论点仅依据本案事实，类案无适用支持。',
            citation_decision='used' if entry and entry['paths'] else 'not_needed',
            citations=[entry['paths'][0]['path_id']] if entry and entry['paths'] else [])],
        responsible_parties=['A公司'],law_articles=[],liability_conclusion='测试结论，不作实际法律判断。')


def fake_semantics(query, documents):
    # Deterministic stand-in; real embedding model quality is evaluated separately.
    return [s / (s + 1) for s in retrieve._bm25(query, documents)]


class SchemaTests(unittest.TestCase):
    def test_same_name_is_local(self):
        self.assertNotEqual(fixture('a')['entities'][0]['entity_id'],fixture('b')['entities'][0]['entity_id'])

    def test_bad_endpoint_rejected(self):
        data=fixture();data['relations'][0]['dst']='missing'
        with self.assertRaises(ValueError):validate_case(data)

    def test_non_source_quote_rejected(self):
        data=fixture();data['causal_paths'][0]['evidence_quotes']=['不存在的原文']
        with self.assertRaises(ValueError):validate_case(data,data['source_text'])

    def test_bad_chain_type_rejected(self):
        data=fixture();data['causal_paths'][0]['node_ids']='bo'
        with self.assertRaises(ValueError):validate_case(data)

    def test_review_metadata_does_not_block_valid_graph_data(self):
        data=fixture();data['_annotation_status']='reviewed'
        self.assertEqual(validate_case(data)['_annotation_status'],'reviewed')
        data['_annotation_status']='draft'
        self.assertEqual(validate_case(data)['_annotation_status'],'draft')

    def test_invalid_citation_rejected(self):
        data=answer();data['reasoning_paths'][0]['citations']=['invented']
        with self.assertRaises(ValueError):generation.validate_answer(data)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        semantic = patch.object(retrieve, 'semantic_scores', side_effect=fake_semantics)
        semantic.start();self.addCleanup(semantic.stop)
        self.docs={'hardware':fixture('hardware','雷达硬件缺陷导致碰撞'),
                   'human':fixture('human','驾驶疲劳超速导致碰撞')}

    def test_same_type_different_facts(self):
        q1=dict(facts='雷达硬件缺陷',question='雷达硬件故障',case_type='交通事故')
        q2=dict(facts='驾驶疲劳超速',question='疲劳驾驶超速',case_type='交通事故')
        a=retrieve.retrieve_memory(q1,self.docs)
        b=retrieve.retrieve_memory(q2,self.docs)
        self.assertEqual(a['ranked_cases'][0]['case_id'],'hardware')
        self.assertEqual(b['ranked_cases'][0]['case_id'],'human')

    def test_self_exclusion(self):
        entry=retrieve.retrieve_memory(dict(facts='雷达',case_type='交通事故',source_case_id='hardware'),self.docs)
        self.assertNotIn('hardware',evaluate.retrieval_ranked_ids(entry))

    def test_backend_consistency_even_if_db_paths_reversed(self):
        data=copy.deepcopy(self.docs['hardware'])
        cp=copy.deepcopy(data['causal_paths'][0]);cp['id']='p2';data['causal_paths'].append(cp)
        self.docs['hardware']=validate_case(data,data['source_text'])
        rows=[]
        for cid,d in reversed(list(self.docs.items())):
            base={k:v for k,v in d.items() if k!='causal_paths'}
            rows.append(dict(case_id=cid,data_json=json.dumps(base),
                             paths=[json.dumps(p) for p in reversed(d['causal_paths'])]))
        session=Mock();session.__enter__=Mock(return_value=session);session.__exit__=Mock(return_value=False)
        session.run.return_value=rows
        driver=Mock();driver.__enter__=Mock(return_value=driver);driver.__exit__=Mock(return_value=False)
        driver.session.return_value=session
        q=dict(facts='雷达硬件缺陷',question='雷达',case_type='交通事故')
        with patch.object(retrieve,'get_driver',return_value=driver):
            graph=retrieve.retrieve_neo4j(q,self.docs)
        self.assertEqual(graph,retrieve.retrieve_memory(q,self.docs))

    def test_backend_mismatch_fails(self):
        session=Mock();session.__enter__=Mock(return_value=session);session.__exit__=Mock(return_value=False)
        session.run.return_value=[]
        driver=Mock();driver.__enter__=Mock(return_value=driver);driver.__exit__=Mock(return_value=False)
        driver.session.return_value=session
        with patch.object(retrieve,'get_driver',return_value=driver):
            with self.assertRaises(RuntimeError):retrieve.retrieve_neo4j(dict(facts='雷达'),self.docs)

    def test_fact_change_invalidates_cache(self):
        a=dict(id='same',facts='雷达');b=dict(a,facts='疲劳')
        self.assertNotEqual(retrieve.retrieval_key(a,self.docs,[],'memory'),
                            retrieve.retrieval_key(b,self.docs,[],'memory'))

    def test_ranks_are_not_derived_from_paths(self):
        self.assertEqual(evaluate.retrieval_ranked_ids(dict(ranked_cases=[dict(case_id='first'),dict(case_id='second')],
                                                         paths=[dict(case_id='second')])),['first','second'])


class LLMTests(unittest.TestCase):
    def setUp(self):
        semantic = patch.object(retrieve, 'semantic_scores', side_effect=fake_semantics)
        semantic.start();self.addCleanup(semantic.stop)

    def test_both_clients_use_explicit_network_policy(self):
        for trust_env in (False,True):
            with patch.object(config,'LLM_TRUST_ENV',trust_env),patch.object(config,'DEEPSEEK_API_KEY','test-generation'),patch.object(config,'ZHIPU_API_KEY','test-judge'),patch.object(llm.httpx,'Client') as http,patch.object(llm,'OpenAI') as api:
                llm.get_client();llm.get_zhipu_client()
                self.assertEqual(http.call_count,2)
                for call in http.call_args_list:
                    self.assertEqual(call.kwargs['trust_env'],trust_env)
                    self.assertNotIn('verify',call.kwargs)
                for call in api.call_args_list:
                    self.assertIs(call.kwargs['http_client'],http.return_value)
                    self.assertEqual(call.kwargs['max_retries'],0)

    def test_connection_diagnostics_show_cause_without_credentials(self):
        request=llm.httpx.Request('POST','https://example.invalid/chat/completions')
        exc=llm.APIConnectionError(request=request)
        exc.__cause__=llm.httpx.ConnectError('TLS EOF with test-secret-value and sk-example-secret')
        with patch.object(config,'DEEPSEEK_API_KEY','test-secret-value'):
            diagnostic=llm.format_error(exc)
        self.assertIn('网络连接失败',diagnostic)
        self.assertIn('TLS EOF',diagnostic)
        self.assertNotIn('test-secret-value',diagnostic)
        self.assertNotIn('sk-example-secret',diagnostic)

    def test_http_error_diagnostics_distinguish_balance_from_network(self):
        request=llm.httpx.Request('POST','https://example.invalid/chat/completions')
        exc=llm.APIStatusError('Insufficient balance',response=llm.httpx.Response(402,request=request),body={})
        self.assertIn('HTTP 402',llm.format_error(exc))
        self.assertIn('余额不足',llm.format_error(exc))

    def test_retry_budget_not_squared(self):
        client=Mock()
        client.chat.completions.create.return_value=SimpleNamespace(choices=[SimpleNamespace(
            finish_reason='stop',message=SimpleNamespace(content='bad json'))],usage=None)
        with patch.object(llm.time,'sleep'),patch.object(config,'LLM_MAX_RETRIES',3):
            with self.assertRaises(ValueError):llm.extract_json([],client=client)
        self.assertEqual(client.chat.completions.create.call_count,3)

    def test_permanent_error_not_retried(self):
        client=Mock();client.chat.completions.create.side_effect=RuntimeError('configuration')
        with self.assertRaises(RuntimeError):llm.extract_json([],client=client)
        self.assertEqual(client.chat.completions.create.call_count,1)

    def test_empty_or_truncated_answer_rejected(self):
        client=Mock();client.chat.completions.create.return_value=SimpleNamespace(choices=[SimpleNamespace(
            finish_reason='length',message=SimpleNamespace(content='partial'))],usage=None)
        with patch.object(llm.time,'sleep'):
            with self.assertRaises(ValueError):llm.chat([],client=client)

    def test_judge_failure_propagates(self):
        with patch.object(llm,'extract_json',side_effect=RuntimeError('offline')):
            with self.assertRaises(RuntimeError):evaluate.judge(dict(facts='事实'),'回答',object())

    def test_judge_receives_facts(self):
        valid=dict(comprehensiveness=7,interpretability=7,accuracy=7,reason='具体理由')
        with patch.object(llm,'extract_json',return_value=valid) as call:
            evaluate.judge(dict(facts='独有事故事实'),'回答',object())
        self.assertIn('独有事故事实',call.call_args.args[0][1]['content'])

    def test_score_range_and_boolean_validated(self):
        for value in (11,-1,True,'7'):
            with self.assertRaises(ValueError):evaluate._score_schema(dict(comprehensiveness=value,interpretability=7,accuracy=7,reason='理由'))

    def test_langchain_retrieval_to_generation(self):
        entry=retrieve.retrieve_memory(dict(facts='硬件缺陷',case_type='交通事故'),{'d':fixture()})
        def fake(messages,validator=None,**kwargs):return validator(answer(entry))
        with patch.object(llm,'extract_json',side_effect=fake):
            out=generation.make_chain(object(),lambda case:entry).invoke(dict(facts='硬件缺陷',question='问题'))
        self.assertEqual(out['retrieval'],entry)
        self.assertIn('P1（',out['answer'])
        self.assertIn(entry['paths'][0]['path_id'],out['structured']['reasoning_paths'][0]['citations'])

    def test_reference_answers_not_in_generation_prompt(self):
        prompt=generation.build_prompt(dict(facts='事实',question='问题',legal_context='明确的适用法时间前提',reference_answer='SECRET GOLD'))
        self.assertNotIn('SECRET GOLD',prompt)
        self.assertIn('明确的适用法时间前提',prompt)


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        semantic = patch.object(retrieve, 'semantic_scores', side_effect=fake_semantics)
        semantic.start();self.addCleanup(semantic.stop)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        values=dict(BASE_DIR=self.root,DOCX_DIR=self.root/'corpus',DATA_DIR=self.root/'data',
            PARSED_DIR=self.root/'data/parsed',EXTRACTED_DIR=self.root/'data/extracted',
            OUTPUT_DIR=self.root/'output',ANNOTATIONS_DIR=self.root/'annotations/cases',
            LAWS_PATH=self.root/'annotations/laws.json',TESTSET_PATH=self.root/'testset.json',
            BASELINE_RESULTS=self.root/'output/baseline_results.json',
            GRAPHRAG_RESULTS=self.root/'output/graphrag_results.json',
            COMPARISON_CSV=self.root/'output/comparison.csv',EVAL_REPORT=self.root/'output/evaluation_report.md')
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        for name,value in values.items():self.stack.enter_context(patch.object(config,name,value))
        write_json(config.LAWS_PATH,dict(articles=[]))
        test=dict(id='q',facts='雷达硬件缺陷导致碰撞',question='雷达硬件缺陷',case_type='交通事故',
            reference_answer='测试标准答案',reference=dict(responsible_parties=['A公司'],law_articles=[],liability_conclusion='测试结论'),
            relevant_case_ids=['document'],_annotation_status='draft')
        write_json(config.TESTSET_PATH,dict(cases=[test]))

    def seed(self):
        d=fixture()
        config.PARSED_DIR.mkdir(parents=True,exist_ok=True)
        (config.PARSED_DIR/'document.txt').write_text(d['source_text'],encoding='utf-8')
        write_json(config.EXTRACTED_DIR/'document.json',d)

    def test_fresh_parse_extract_generate_evaluate_and_resume(self):
        from docx import Document
        testset=read_json(config.TESTSET_PATH)
        testset['cases'][0]['legal_context']='法律核对时点2026-10-26'
        write_json(config.TESTSET_PATH,testset)
        config.DOCX_DIR.mkdir()
        doc=Document();doc.add_paragraph('硬件缺陷导致碰撞');doc.save(str(config.DOCX_DIR/'document.docx'))
        raw=fixture(reviewed=True)
        def fake(messages,validator=None,**kwargs):
            data=raw if '裁判' in messages[1]['content'] or messages[0]['content']==extract_kg.SYSTEM_PROMPT else answer()
            # Generation has no citations here, an allowed choice when giving a matching explanation.
            return validator(copy.deepcopy(data)) if validator else copy.deepcopy(data)
        def graph_retrieval(case,extracted,top_k,laws,paths_per_case):
            return retrieve.retrieve_memory(case,extracted,top_k,laws,paths_per_case)
        with patch.object(llm,'get_client',return_value=object()),patch.object(llm,'extract_json',side_effect=fake) as calls,patch.object(build_graph,'main'),patch.object(retrieve,'retrieve_neo4j',side_effect=graph_retrieval),patch.object(llm,'get_zhipu_client',side_effect=RuntimeError('offline test')):
            run.main()
            self.assertEqual(read_json(config.EXTRACTED_DIR/'document.json')['_annotation_status'],'draft')
            self.assertNotIn('reviewed_by',read_json(config.EXTRACTED_DIR/'document.json'))
            self.assertEqual(read_json(config.BASELINE_RESULTS)['q']['input']['legal_context'],'法律核对时点2026-10-26')
            count=calls.call_count
            run.main()
            self.assertEqual(calls.call_count,count)
            with patch.object(config,'HYBRID_RRF_K',config.HYBRID_RRF_K+1):
                run.main()
            # Only GraphRAG generation changes; extraction and baseline remain reusable.
            self.assertEqual(calls.call_count,count+1)
        result=evaluate.main(offline=True)
        record=result['cases']['q']
        self.assertTrue(record['comparable'])
        self.assertNotIn('accuracy',record['metrics']['baseline'])
        self.assertEqual(record['metrics']['baseline']['party_f1'],1)
        self.assertEqual(record['retrieval']['mrr'],1)
        self.assertTrue(config.EVAL_REPORT.exists())

    def test_review_overlay_and_stale_source(self):
        self.seed()
        reviewed=fixture(reviewed=True)
        write_json(config.ANNOTATIONS_DIR/'document.json',reviewed)
        self.assertEqual(len(annotations.load_cases()),1)
        (config.PARSED_DIR/'document.txt').write_text('改变原文',encoding='utf-8')
        with self.assertRaises(ValueError):annotations.load_cases()

    def test_api_outage_remains_missing_in_report(self):
        inputs={k:read_json(config.TESTSET_PATH)['cases'][0].get(k,'') for k in ('id','facts','question','case_type')}
        record=dict(status='ok',input=inputs,model='same',structured=answer(),answer='测试回答')
        write_json(config.BASELINE_RESULTS,{'q':record});write_json(config.GRAPHRAG_RESULTS,{'q':record})
        with patch.object(llm,'get_zhipu_client',return_value=object()),patch.object(llm,'extract_json',side_effect=RuntimeError('outage')),patch.object(llm,'chat',side_effect=RuntimeError('outage')):
            result=evaluate.main()
        r=result['cases']['q']
        self.assertNotIn('accuracy',r['metrics']['baseline'])
        self.assertNotIn('conclusion_acc',r['metrics']['baseline'])
        self.assertEqual(r['operations']['baseline_judge']['status'],'error')

    def test_plain_entrypoint_runs_all_stages_with_neo4j(self):
        with ExitStack() as stack:
            calls={name:stack.enter_context(patch.object(module,'main')) for name,module in
                [('parse',docx_parser),('extract',extract_kg),('build',build_graph),
                 ('baseline',baseline),('reasoning',reasoning),('evaluate',evaluate)]}
            run.main()
        calls['build'].assert_called_once_with()
        calls['reasoning'].assert_called_once_with(backend='neo4j')
        calls['evaluate'].assert_called_once_with()

    def test_build_case_uses_local_identity_and_path_nodes(self):
        tx=Mock();d=fixture()
        build_graph.build_case(tx,d)
        queries=[c.args[0] for c in tx.run.call_args_list]
        self.assertFalse(any('MERGE (n:Party {name:' in q for q in queries))
        self.assertTrue(any('CREATE (p:ADCRCausalPath' in q for q in queries))
        self.assertTrue(any('[r:LAW_RESOURCE]->() DELETE r' in q for q in queries))

    def test_clean_keeps_inputs_and_empty_directories(self):
        self.seed()
        config.DOCX_DIR.mkdir()
        original=config.DOCX_DIR/'original.docx';original.write_bytes(b'original')
        secret=self.root/'.env';secret.write_text('test-only',encoding='utf-8')
        write_json(config.OUTPUT_DIR/'result.json',{'old':True})
        write_json(config.ANNOTATIONS_DIR/'human.json',{'human':True})
        nested_cache=self.root/'adcr/common/__pycache__'
        nested_cache.mkdir(parents=True)
        (nested_cache/'module.pyc').write_bytes(b'cache')
        clean_outputs.clean(apply=True,root=self.root)
        self.assertFalse(nested_cache.exists())
        self.assertEqual(list(config.DATA_DIR.iterdir()),[])
        self.assertEqual(list(config.OUTPUT_DIR.iterdir()),[])
        self.assertTrue(original.exists());self.assertTrue(secret.exists())
        self.assertTrue((config.ANNOTATIONS_DIR/'human.json').exists())

    def test_all_cases_evaluation_resume_without_new_calls(self):
        case=read_json(config.TESTSET_PATH)['cases'][0]
        second=dict(case,id='q2')
        write_json(config.TESTSET_PATH,dict(cases=[case,second]))
        outputs={}
        for c in (case,second):
            inputs={k:c.get(k,'') for k in ('id','facts','question','case_type')}
            outputs[c['id']]=dict(status='ok',input=inputs,model='same',structured=answer(),answer='测试回答')
        write_json(config.BASELINE_RESULTS,outputs);write_json(config.GRAPHRAG_RESULTS,outputs)
        def fake(messages,validator=None,**kwargs):
            content=messages[1]['content']
            data=dict(consistent=True) if 'consistent' in content else dict(comprehensiveness=7,interpretability=8,accuracy=6,reason='测试依据')
            return validator(data)
        with patch.object(llm,'get_zhipu_client',return_value=object()),patch.object(llm,'extract_json',side_effect=fake) as calls,patch.object(llm,'chat',return_value='测试匿名评述') as chat_calls:
            evaluate.main();count=calls.call_count;reviews=chat_calls.call_count
            evaluate.main()
            self.assertEqual(calls.call_count,count)
            self.assertEqual(chat_calls.call_count,reviews)
            self.assertEqual(count,8)

    def test_changed_facts_exclude_old_generation(self):
        case=read_json(config.TESTSET_PATH)['cases'][0]
        inputs={k:case.get(k,'') for k in ('id','facts','question','case_type')}
        inputs['facts']='old facts'
        record=dict(status='ok',input=inputs,model='same',structured=answer(),answer='old answer')
        write_json(config.BASELINE_RESULTS,{'q':record});write_json(config.GRAPHRAG_RESULTS,{'q':record})
        r=evaluate.main(offline=True)['cases']['q']
        self.assertFalse(r['comparable']);self.assertEqual(r['metrics']['baseline'],{})

    def test_build_main_uses_transaction_boundary(self):
        self.seed()
        session=Mock();session.__enter__=Mock(return_value=session);session.__exit__=Mock(return_value=False)
        driver=Mock();driver.__enter__=Mock(return_value=driver);driver.__exit__=Mock(return_value=False)
        driver.session.return_value=session
        with patch.object(build_graph,'get_driver',return_value=driver):build_graph.main()
        self.assertEqual(session.execute_write.call_count,1)
        self.assertIs(session.execute_write.call_args.args[0],build_graph.build_case)

    def test_law_structure_required_but_review_metadata_optional(self):
        law=dict(id='test',name='测试条文',text='测试原文',version='测试版本',source_url='https://example.invalid/test',_annotation_status='reviewed')
        write_json(config.LAWS_PATH,dict(articles=[law]))
        self.assertEqual(annotations.load_laws(),[law])
        del law['source_url']
        write_json(config.LAWS_PATH,dict(articles=[law]))
        with self.assertRaises(ValueError):annotations.load_laws()


if __name__=='__main__':unittest.main()
