"""Offline checks for hybrid recall, fusion, cache isolation and token windows."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from adcr import config
from adcr.retrieval import embeddings
from adcr.retrieval import retrieve
from tests.test_pipeline import fixture


class HybridTests(unittest.TestCase):
    def test_semantic_only_candidate_is_recalled(self):
        docs = {'related': fixture('related', '制动装置失灵引发追尾'),
                'keyword': fixture('keyword', '刹车踏板颜色争议')}
        query = dict(facts='刹车失效', question='', case_type='')
        def semantics(q, texts):
            return [0.95 if '制动装置' in text else 0.05 for text in texts]
        with patch.object(retrieve, 'semantic_scores', side_effect=semantics):
            result = retrieve.retrieve_memory(query, docs)
        related = next(c for c in result['ranked_cases'] if c['case_id'] == 'related')
        self.assertEqual(related['bm25_score'], 0)
        self.assertIsNone(related['keyword_rank'])
        self.assertEqual(related['semantic_rank'], 1)
        self.assertTrue(any(p['case_id'] == 'related' for p in result['paths']))

    def test_rrf_union_and_exact_fusion(self):
        with patch.object(retrieve, '_bm25', return_value=[4, 0, 0]), \
             patch.object(retrieve, 'semantic_scores', return_value=[0.1, 0.9, 0.2]), \
             patch.object(config, 'HYBRID_KEYWORD_WEIGHT', 0.5), \
             patch.object(config, 'HYBRID_SEMANTIC_WEIGHT', 0.5), \
             patch.object(config, 'HYBRID_RRF_K', 60):
            fused = retrieve._hybrid('q', ['a', 'b', 'c'], ['a', 'b', 'c'], 1)
        self.assertEqual({i for i,_ in fused}, {0, 1})
        self.assertAlmostEqual(fused[0][1]['fusion_score'], 0.5/61)

    def test_exclusions_apply_before_semantic_retrieval(self):
        docs = {'self': fixture('self'), 'event': fixture('event'), 'allowed': fixture('allowed')}
        docs['event']['event_group'] = 'same-event'
        query = dict(facts='硬件缺陷', source_case_id='self', event_group='same-event')
        def semantics(q, texts):
            self.assertFalse(any('self' in t or 'event' in t for t in texts))
            return [0.8]*len(texts)
        with patch.object(retrieve, 'semantic_scores', side_effect=semantics):
            result = retrieve.retrieve_memory(query, docs)
        self.assertEqual([c['case_id'] for c in result['ranked_cases']], ['allowed'])

    def test_config_changes_invalidate_retrieval(self):
        docs = {'d': fixture('d')}
        key = retrieve.retrieval_key(dict(facts='事实'), docs, [], 'memory')
        for name, value in [('EMBEDDING_MODEL', 'other'), ('EMBEDDING_REVISION', 'commit'),
                            ('EMBEDDING_QUERY_PREFIX', ''), ('EMBEDDING_CHUNK_TOKENS', 128),
                            ('HYBRID_KEYWORD_WEIGHT', 0.7), ('HYBRID_CANDIDATES', 20)]:
            with patch.object(config, name, value):
                self.assertNotEqual(key, retrieve.retrieval_key(dict(facts='事实'), docs, [], 'memory'))

    def test_semantic_failure_is_not_silently_keyword_only(self):
        with patch.object(retrieve, 'semantic_scores', side_effect=RuntimeError('model unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'model unavailable'):
                retrieve.retrieve_memory(dict(facts='缺陷'), {'d': fixture()})

    def test_no_candidates_does_not_load_model(self):
        with patch.object(embeddings, '_load_model') as load:
            result = retrieve.retrieve_memory(dict(facts='缺陷', source_case_id='d'), {'d': fixture()})
        self.assertEqual(result['ranked_cases'], [])
        load.assert_not_called()

    def test_auto_backend_does_not_hide_semantic_error_as_database_failure(self):
        with patch.object(retrieve, 'retrieve_neo4j', side_effect=retrieve.RetrievalScoringError('model')), \
             patch.object(retrieve, 'retrieve_memory') as memory:
            with self.assertRaises(retrieve.RetrievalScoringError):
                retrieve.retrieve(dict(facts='缺陷'), extracted={'d': fixture()}, laws=[])
        memory.assert_not_called()


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(chr(i) for i in ids)

    def num_special_tokens_to_add(self, pair=False):
        return 2


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.model = SimpleNamespace(tokenizer=CharacterTokenizer(), max_seq_length=8,
            encode=Mock(side_effect=lambda texts, **kw: SimpleNamespace(tolist=lambda: [[1., 0.] for t in texts])))
        for name,value in [('DATA_DIR', Path(self.temp.name)), ('EMBEDDING_CHUNK_TOKENS', 4),
                           ('EMBEDDING_CHUNK_OVERLAP', 1), ('EMBEDDING_QUERY_PREFIX', 'Q')]:
            p = patch.object(config, name, value);p.start();self.addCleanup(p.stop)
        p = patch.object(embeddings, '_load_model', return_value=self.model)
        p.start();self.addCleanup(p.stop)

    def test_tail_covered_and_cached_and_query_instruction_separated(self):
        text = 'abcdefghij'
        embeddings.encode_texts([text])
        chunks = self.model.encode.call_args.args[0]
        self.assertEqual(chunks, ['abcd', 'defg', 'ghij'])
        embeddings.encode_texts([text])
        self.assertEqual(self.model.encode.call_count, 1)
        embeddings.encode_texts([text], query=True)
        self.assertTrue(all(c.startswith('Q') for c in self.model.encode.call_args.args[0]))
        with patch.object(config, 'EMBEDDING_REVISION', 'different'):
            embeddings.encode_texts([text])
        self.assertEqual(self.model.encode.call_count, 3)

    def test_chunk_max_then_query_mean(self):
        with patch.object(embeddings, 'encode_texts', side_effect=[
                [[[1., 0.], [0., 1.]]], [[[1., 0.], [0., 1.]], [[1., 0.]]]]):
            self.assertEqual(embeddings.semantic_scores('q', ['a', 'b']), [1., .5])

    def test_invalid_vectors_rejected(self):
        for vectors in ([[float('nan'), 0.]], [[0., 0.]], [[1., 0.], [1.]]):
            self.model.encode.side_effect = None
            self.model.encode.return_value = SimpleNamespace(tolist=lambda: vectors)
            with self.assertRaises(ValueError):
                embeddings.encode_texts(['text'])


if __name__ == '__main__':
    unittest.main()
