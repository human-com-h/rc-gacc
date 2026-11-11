"""Candidate alias consistency and strict unknown handling for semantic grading."""
import json
import unittest
from rc_gacc.gacc_closeout import METHODS
from rc_gacc.qa_judging import build_packets, parse_verdict, summarize_labels, paired_label_bounds


class PackedJudgeTests(unittest.TestCase):
    def fixture(self):
        question = dict(goal_id='g',question_id='q',context_id='context',question='What?',gold_answer='Reference')
        mapped = [dict(question,method=m,raw_response='answer' if i<3 else 'alternative',
                       request_id='r'+str(i),hit_generation_cap=False) for i,m in enumerate(METHODS)]
        return question,mapped

    def test_same_answers_share_one_candidate_and_messages_hide_methods(self):
        question,mapped = self.fixture()
        packets,aliases = build_packets(mapped,[question])
        self.assertEqual(len(packets),1)
        self.assertEqual(len(packets[0]['candidate_ids']),2)
        self.assertEqual(len(aliases),5)
        self.assertEqual(aliases[0]['candidate_id'],aliases[1]['candidate_id'])
        body = json.loads(packets[0]['messages'][1]['content'].split('\nReturn')[0])
        self.assertEqual(set(body),{'question','reference_answer','candidates'})
        self.assertTrue(all(set(c)=={'candidate_id','answer'} for c in body['candidates']))
        self.assertFalse(any(m in str(packets[0]['messages']) for m in METHODS))

    def test_candidate_order_independent_of_method_iteration(self):
        question,mapped = self.fixture()
        a,_ = build_packets(mapped,[question])
        b,_ = build_packets(list(reversed(mapped)),[question])
        self.assertEqual(a,b)

    def test_incomplete_or_inconsistent_method_groups_fail(self):
        question,mapped = self.fixture()
        with self.assertRaises(ValueError):build_packets(mapped[:-1],[question])
        mapped[0]['gold_answer']='Different'
        with self.assertRaises(ValueError):build_packets(mapped,[question])

    def test_complete_three_valued_verdict(self):
        p = parse_verdict('{"c0":"correct","c1":"incorrect","c2":"unknown"}',['c0','c1','c2'])
        self.assertTrue(p['valid'])
        self.assertEqual(p['verdicts']['c2'],'unknown')

    def test_malformed_packets_never_silently_grade_missing_candidates_wrong(self):
        for raw in ['{}','{"c0":"correct"}', '{"c0":"correct","c1":true}',
                    '{"c0":"correct","c1":{}}','{"c0":"correct","c1":"wrong"}',
                    '```json\n{"c0":"correct","c1":"incorrect"}\n```',
                    '{"c0":"correct","c0":"incorrect","c1":"correct"}']:
            with self.subTest(raw=raw):
                parsed = parse_verdict(raw,['c0','c1'])
                self.assertFalse(parsed['valid'])
                self.assertEqual(parsed['verdicts'],dict(c0='unknown',c1='unknown'))

    def test_unknown_cannot_rescue_a_goal_with_a_known_error(self):
        rows=[dict(goal_id='g',context_id='s',label=x,hit_generation_cap=False) for x in ['correct','incorrect','unknown']]
        summary=summarize_labels(rows)
        self.assertEqual((summary['micro_lower'],summary['micro_upper']),(1/3,2/3))
        self.assertEqual((summary['macro_lower'],summary['macro_upper']),(0,0))
        rows[1]['label']='correct'
        summary=summarize_labels(rows)
        self.assertEqual((summary['macro_lower'],summary['macro_upper']),(0,1))

    def test_shared_unknown_candidate_cancels_in_paired_difference(self):
        a=dict(packet_id='p',candidate_id='c0',label='unknown')
        self.assertEqual(paired_label_bounds(a,a),(0,0))
        b=dict(packet_id='p',candidate_id='c1',label='unknown')
        self.assertEqual(paired_label_bounds(a,b),(-1,1))
        b['label']='correct'
        self.assertEqual(paired_label_bounds(a,b),(-1,0))

    def test_conflicting_labels_for_same_candidate_are_rejected(self):
        a=dict(packet_id='p',candidate_id='c0',label='correct')
        b=dict(a,label='incorrect')
        with self.assertRaises(ValueError):paired_label_bounds(a,b)


if __name__=='__main__':
    unittest.main()
