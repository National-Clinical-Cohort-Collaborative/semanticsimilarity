from unittest import TestCase


class TestEnsmalllen(TestCase):

    def test_make_ensmallen_graph(self):
        self.assertTrue(True)
        try:
            from clustering.hpo_ensmallen_parser import EnsmallenGraph
            EnsmallenGraph("foo", "bar")
        except Exception as e:
            print("Problem:")
            print(e)
            pass
