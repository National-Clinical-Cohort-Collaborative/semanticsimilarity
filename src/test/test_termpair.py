from unittest import TestCase
from clustering.term_pair import TermPair



class TestTermPair(TestCase):

    def test_equality(self):
        hpoA = 'HP:0000001'
        hpoB = 'HP:0000002'

        tp1 = TermPair(termA=hpoA, termB=hpoB)
        tp2 = TermPair(termA=hpoA, termB=hpoB)
        self.assertEqual(tp1, tp2)

    def test_two_orders(self):
        hpoA = 'HP:0000001'
        hpoB = 'HP:0000002'

        tp1 = TermPair(termA=hpoA, termB=hpoB)
        tp2 = TermPair(termA=hpoB, termB=hpoA)
        self.assertEqual(tp1, tp2)

    def test_inequality(self):
        hpoA = 'HP:0000001'
        hpoB = 'HP:0000002'
        hpoC = 'HP:0000003'

        tp1 = TermPair(termA=hpoA, termB=hpoB)
        tp2 = TermPair(termA=hpoA, termB=hpoC)
        self.assertNotEqual(tp1, tp2)
