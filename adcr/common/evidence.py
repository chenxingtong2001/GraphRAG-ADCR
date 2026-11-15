"""Shared evidence projection from the immutable snapshot saved with an answer."""
import collections


def evidence_paths(entry):
    fields = ('path_id', 'case_id', 'case_name', 'source_file', 'chain', 'causal_type',
              'causal_pattern', 'explanation', 'annotation_status')
    return [dict({k: p.get(k, '未记录') for k in fields},
                 evidence_quotes=[q[:600] for q in p.get('evidence_quotes', [])[:2]])
            for p in (entry or {}).get('paths', [])]


def citation_audit(structured, entry):
    """Audit citation IDs against retrieved paths."""
    allowed = {p['path_id'] for p in (entry or {}).get('paths', [])}
    match = structured.get('case_matching_citations', [])
    reasoning = [c for p in structured.get('reasoning_paths', []) for c in p.get('citations', [])]
    all_ids = match + reasoning
    decisions = collections.Counter(p.get('citation_decision', '未记录')
                                    for p in structured.get('reasoning_paths', []))
    return dict(invalid_citations=[c for c in all_ids if c not in allowed],
                reasoning_citation_count=len(reasoning),
                unique_reasoning_citation_count=len(set(reasoning)),
                case_matching_citation_count=len(match),
                explicit_rejection_count=decisions['rejected'],
                unrecorded_decision_count=decisions['未记录'],
                snapshot_available=entry is not None,
                existence_does_not_establish_support=True)


def evaluation_evidence(result):
    result = result or {}
    entry = result.get('retrieval')
    structured = result.get('structured') or {}
    return dict(snapshot_status='recorded' if entry is not None else 'not_recorded',
                paths=evidence_paths(entry),
                case_matching_citations=structured.get('case_matching_citations', '未记录'),
                reasoning_citations=[dict(chain=p.get('chain', []), citations=p.get('citations', []),
                    citation_note=p.get('citation_note', '未记录'),
                    citation_decision=p.get('citation_decision', '未记录'))
                    for p in structured.get('reasoning_paths', [])],
                citation_audit=citation_audit(structured, entry))
