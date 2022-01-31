from unittest import TestCase
from clustering.hpo_ensmallen_parser import Hpo2EnsmallenParser
from clustering.hpo_ensmallen import HpoEnsmallen
from parameterized import parameterized
import os


class TestEnsmallen(TestCase):

    @classmethod
    def setUpClass(cls):
        # The following gets us the directory of this file
        dir = os.path.dirname(os.path.abspath(__file__))
        cls.path = os.path.join(dir, "test_data/test_hpo_graph.tsv")

    def test_file_is_found(self):
        self.assertTrue(os.path.isfile(self.path))

    def test_ctor_ensmaller_parser(self):
        p = make_ensmallen_graph_parser_object(self.path)
        self.assertIsInstance(p, Hpo2EnsmallenParser)

    def test_graph(self):
        parser = Hpo2EnsmallenParser(self.path)
        self.assertTrue(parser)
        self.assertTrue(isinstance(parser, Hpo2EnsmallenParser))
        g = parser.graph
        self.assertTrue(g)

    def test_get_ancestors(self):
        go = HpoEnsmallen(self.path)
        self.assertTrue(hasattr(go, 'get_ancestors'))
        anc = go.get_ancestors('HP:0012638')
        self.assertCountEqual(anc, ['HP:0012638', 'HP:0000707', 'HP:0000118'])

    def test_get_descendents(self):
        go = HpoEnsmallen(self.path)
        self.assertTrue(hasattr(go, 'get_descendents'))
        anc = go.get_descendents('HP:0000118')
        self.assertCountEqual(anc, ['HP:0000118', 'HP:0000707', 'HP:0000818', 'HP:0000834', 'HP:0000873', 'HP:0003549',
                                    'HP:0009025', 'HP:0009124', 'HP:0012638', 'HP:0012639', 'HP:0100568',
                                    'HP:0100881', 'HP:0410008'])

    # @parameterized.expand([
    #     ['HP:PARENT', ['HP:CHILD1','HP:CHILD2', 'HP:GRANDCHILD1'], []],
    #     ['HP:CHILD1', [], ['HP:PARENT']],
    #     ['HP:CHILD2', ['HP:GRANDCHILD1'], ['HP:PARENT']],
    #     ['HP:GRANDCHILD1', [], ['HP:CHILD2', 'HP:PARENT']],
    # ])
    # def test_simple_graph(self, node, descendents, ancestors):
    #     ensmallen = HpoEnsmallen(hpo_graph=self.hpo_path_tiny)
    #     pass


def make_ensmallen_graph_parser_object(hpo_graph):
    try:
        parser = Hpo2EnsmallenParser(hpo_graph)
        return parser
    except Exception as e:  # try/exception here is necessary to get a useful error message
        print("Problem making :")
        print(e)
        return None

