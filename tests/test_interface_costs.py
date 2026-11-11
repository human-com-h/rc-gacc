import itertools
import unittest
import numpy as np
from rc_gacc.selector import Edge,solve_milp,objective
from rc_gacc.epistemic_interface import render

class InterfaceCostsTests(unittest.TestCase):
    def test_edge_register_is_in_exact_discrete_budget(self):
        costs=[3,4,3,5,2];u=np.array([.9,.8,.7,.6,.2]);a=np.array([[.8,.9,.2,.5,.1]])
        edges=[Edge(0,1,'C'),Edge(2,3,'Q')];edge_costs=[5,3]
        selected,log=solve_milp(costs,13,u,a,edges,edge_costs=edge_costs)
        values=[]
        for bits in itertools.product([0,1],repeat=5):
            charge=sum(c*b for c,b in zip(costs,bits))+5*bits[0]*bits[1]+3*bits[2]*bits[3]
            if charge<=13 and bits[0]==bits[1] and bits[2]<=bits[3]:
                values.append(objective([i for i,b in enumerate(bits) if b],u,a,edges))
        self.assertAlmostEqual(objective(selected,u,a,edges),max(values),places=6)
        self.assertLessEqual(log['selected_proxy_cost'],13)

    def test_register_has_direction_and_no_orphan_claim(self):
        units=[{'sid':7,'text':'Claim.'},{'sid':9,'text':'Only under condition Z.'}]
        e=[Edge(0,1,'Q')]
        full=render(units,[0,1],e,'structured')
        self.assertIn('[SCOPE_LIMIT claim=s7 qualifier=s9]',full)
        orphan=render(units,[0],e,'structured')
        self.assertNotIn('SCOPE_LIMIT',orphan)
        self.assertNotIn('CERTIFICATE',orphan)

if __name__=='__main__':unittest.main()
