"""Regression checks for evidence visibility, fair scoring, and citation provenance."""
import json
import unittest
from unittest.mock import patch

from adcr.evaluation import evaluate
from adcr.generation import generate as generation
from adcr.common.evidence import citation_audit, evaluation_evidence, evidence_paths
from adcr.common.storage import fingerprint, read_json, write_json
from tests import test_pipeline
from tests.test_pipeline import answer, fixture


def snapshot():
    d=fixture('source')
    path=dict(d['causal_paths'][0],case_id='source',case_name='实际参考案件',source_file='原文.txt',annotation_status='draft')
    return dict(paths=[path],ranked_cases=[dict(case_id='source',case_name='实际参考案件',rank=1,similarity=.5,matched_terms=[])],
                laws=[],_fingerprint='snapshot',retrieval_version='fixed')


class CitationTests(unittest.TestCase):
    def test_shared_projection_limits_are_identical(self):
        entry=snapshot();entry['paths'][0]['evidence_quotes']=['甲'*900,'乙'*700,'不应发送']
        projected=evidence_paths(entry)
        self.assertEqual([len(q) for q in projected[0]['evidence_quotes']],[600,600])
        self.assertEqual(evaluation_evidence(dict(retrieval=entry))['paths'],projected)
        prompt=generation.format_paths(entry['paths'])
        self.assertNotIn('甲'*601,prompt);self.assertNotIn('不应发送',prompt)

    def test_real_id_and_fabricated_id_have_distinct_audits(self):
        entry=snapshot();data=answer(entry)
        self.assertEqual(citation_audit(data,entry)['invalid_citations'],[])
        data['case_matching_citations']=['invented']
        self.assertEqual(citation_audit(data,entry)['invalid_citations'],['invented'])
        with self.assertRaises(ValueError):generation.validate_answer(data,entry)

    def test_existing_source_is_not_automatically_support(self):
        entry=snapshot();data=answer(entry)
        data['reasoning_paths'][0].update(citations=[],citation_decision='rejected',
            citation_note='此案只涉及硬件缺陷，无法支持本案租赁合同论点，因此拒用。')
        data['case_matching_citations']=[entry['paths'][0]['path_id']]
        generation.validate_answer(data,entry)
        audit=citation_audit(data,entry)
        self.assertEqual(audit['invalid_citations'],[])
        self.assertEqual(audit['reasoning_citation_count'],0)
        self.assertEqual(audit['explicit_rejection_count'],1)
        self.assertTrue(audit['existence_does_not_establish_support'])
        self.assertIn('来源存在也不等于支持论点',evaluate.EVIDENCE_RULES)

    def test_graph_answer_requires_citation_fields(self):
        entry=snapshot();data=answer(entry)
        del data['case_matching_citations']
        with self.assertRaises(ValueError):generation.validate_answer(data,entry)
        data=answer(entry);del data['reasoning_paths'][0]['citation_note']
        with self.assertRaises(ValueError):generation.validate_answer(data,entry)

    def test_citation_note_required_even_without_reference(self):
        entry=snapshot();data=answer()
        data['reasoning_paths'][0]['citation_note']=''
        with self.assertRaises(ValueError):generation.validate_answer(data,entry)
        data=answer(entry);data['reasoning_paths'][0]['citation_decision']='rejected'
        with self.assertRaises(ValueError):generation.validate_answer(data,entry)

    def test_readable_source_table_and_full_id_kept_in_structure(self):
        entry=snapshot();data=answer(entry);pid=entry['paths'][0]['path_id']
        rendered=generation.render_answer(data,entry)
        self.assertIn('P1（实际参考案件）',rendered)
        self.assertIn('原文.txt',rendered)
        self.assertIn(entry['paths'][0]['evidence_quotes'][0],rendered)
        self.assertNotIn(pid,rendered)
        self.assertEqual(data['reasoning_paths'][0]['citations'],[pid])

    def test_judge_gets_evidence_and_missing_snapshot_is_not_fabrication(self):
        evidence=evaluation_evidence(dict(retrieval=snapshot(),structured=answer()))
        with patch.object(evaluate.llm,'extract_json',return_value={}) as call:
            evaluate.judge(dict(facts='当前案件'), '回答',object(),evidence=evidence)
        prompt=call.call_args.args[0][1]['content']
        self.assertIn('实际参考案件',prompt);self.assertIn('原文.txt',prompt)
        self.assertIn('没有类案、未引用类案不自动扣分',prompt)
        self.assertIn('snapshot_status=not_recorded',prompt)

    def test_anonymous_evidence_follows_both_orderings(self):
        for reverse in (False,True):
            cid=next(str(i) for i in range(100) if (int(fingerprint(str(i))[:8],16)%2==1)==reverse)
            with patch.object(evaluate.llm,'chat',return_value='比较') as call:
                evaluate.qualitative_review(dict(id=cid),'base','graph',object(),{'marker':'base-evidence'},{'marker':'graph-evidence'})
            prompt=call.call_args.args[0][1]['content']
            payload=json.loads(prompt.split('\n')[-1])
            for side in ('A','B'):
                self.assertEqual(payload['evidence_for_verification_'+side]['marker'],payload['answer_'+side]+'-evidence')

    def test_baseline_prompt_contains_task_and_output_format(self):
        case=dict(facts='事实',question='问题')
        self.assertEqual(generation.build_prompt(case),'【案件事实】\n事实\n【待回答问题】\n问题\n\n'+generation.OUTPUT_SPEC)


class EvidencePipelineTests(unittest.TestCase):
    setUp=test_pipeline.EndToEndTests.setUp

    def test_evaluation_uses_saved_snapshot_and_invalidates_changed_evidence(self):
        case=read_json(self.root/'testset.json')['cases'][0]
        inputs={k:case.get(k,'') for k in ('id','facts','question','case_type')}
        base=dict(status='ok',input=inputs,model='same',structured=answer(),answer='固定回答')
        graph=dict(base,retrieval=snapshot())
        write_json(self.root/'output/baseline_results.json',{'q':base})
        write_json(self.root/'output/graphrag_results.json',{'q':graph})
        write_json(self.root/'output/retrieval_cache.json',{'q':dict(paths=[{'case_name':'错误的新缓存'}])})
        def fake(messages,validator=None,**kwargs):
            content=messages[1]['content']
            result=dict(consistent=True) if 'consistent' in content else dict(comprehensiveness=7,interpretability=7,accuracy=7,reason='按三方面评估')
            return validator(result)
        with patch.object(evaluate.llm,'get_zhipu_client',return_value=object()), \
             patch.object(evaluate.llm,'extract_json',side_effect=fake) as calls, \
             patch.object(evaluate.llm,'chat',return_value='比较') as chats:
            d=evaluate.main();count=calls.call_count
            self.assertEqual(d['cases']['q']['evidence']['graphrag']['paths'][0]['case_name'],'实际参考案件')
            self.assertFalse(any('错误的新缓存' in str(c) for c in calls.call_args_list))
            evaluate.main();self.assertEqual(calls.call_count,count)
            graph['retrieval']['paths'][0]['explanation']='证据解释有变'
            write_json(self.root/'output/graphrag_results.json',{'q':graph})
            evaluate.main()
            self.assertEqual(calls.call_count,count+1)
            self.assertEqual(chats.call_count,2)
