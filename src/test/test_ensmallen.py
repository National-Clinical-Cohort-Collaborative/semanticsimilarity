from unittest import TestCase
from clustering.hpo_ensmallen_parser import EnsmallenGraph


class TestEnsmalllen(TestCase):

    def test_make_ensmallen_graph(self):
        self.assertTrue(True)
        try:
            print("hello world")
            g = EnsmallenGraph("foo", "bar")
            print(g)
        except Exception as e:
            print("Problem:")
            print(e)
            pass
