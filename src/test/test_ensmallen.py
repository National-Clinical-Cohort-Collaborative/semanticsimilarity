from unittest import TestCase
from clustering.hpo_ensmallen_parser import Hpo2EnsmallenParser
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
        try:
            parser = Hpo2EnsmallenParser(self.path)
            self.assertTrue(not False)
        except Exception as e:  # try/exception here is necessary to get a useful error message
            print("Problem making :")
            print(e)
            self.assertTrue(False, msg="Problem creating Hpo2EnsmallenParser instance")

    def test_graph(self):
        parser = Hpo2EnsmallenParser(self.path)
        self.assertTrue(parser)
        self.assertTrue(isinstance(parser, Hpo2EnsmallenParser))
        g = parser.graph
        self.assertTrue(g)
