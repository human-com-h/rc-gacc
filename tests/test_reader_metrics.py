import unittest
from rc_gacc.reader_metrics import aggregate

class ReaderMetricsTests(unittest.TestCase):
    def plan(self):
        return [dict(family='keep_semantic',id='c',method='m',position=0,track='keep_conditional',judge='key'),
                dict(family='keep_semantic',id='c',method='m',position=1,track='keep_conditional',judge='proposition')]

    def test_qualifier_is_required_even_when_proposition_yes(self):
        e=self.plan();p=[dict(e[0],parsed={'condition_judgement':'no','overall_judgement':'yes'}),
                         dict(e[1],parsed={'judgement':'yes'})]
        cases,summary=aggregate(e,p)
        self.assertFalse(cases[0]['all_rec']);self.assertFalse(cases[0]['cond_rec'])
        self.assertEqual(summary[0]['complete_valid_cases'],1)

    def test_missing_judgement_is_unscorable(self):
        e=self.plan();cases,summary=aggregate(e,[dict(e[1],parsed={'judgement':'yes'})])
        self.assertIsNone(cases[0]['all_rec']);self.assertEqual(summary[0]['complete_valid_cases'],0)

    def test_duplicate_output_is_rejected(self):
        e=self.plan();row=dict(e[0],parsed={'condition_judgement':'yes','overall_judgement':'yes'})
        with self.assertRaises(ValueError):aggregate(e,[row,row])

    def test_directional_commitment_is_separate_from_unknown(self):
        e=[dict(family='source_relation_reader',id='g_C',method='m',position=0,track='source_anchored_C',
                labels=['AFFIRMED','NEGATED','CONFLICT','UNKNOWN'],gold='CONFLICT')]
        c,_=aggregate(e,[dict(e[0],parsed={'answer':'UNKNOWN'})])
        self.assertFalse(c[0]['correct']);self.assertFalse(c[0]['false_directional'])
        c,_=aggregate(e,[dict(e[0],parsed={'answer':'AFFIRMED'})])
        self.assertFalse(c[0]['correct']);self.assertTrue(c[0]['false_directional'])

if __name__=='__main__':unittest.main()
