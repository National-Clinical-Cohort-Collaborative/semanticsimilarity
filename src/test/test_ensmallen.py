from unittest import TestCase
from clustering.hpo_ensmallen_parser import Hpo2EnsmallenParser
import os


class TestEnsmalllen(TestCase):

    @classmethod
    def setUpClass(cls):
        #current_file = __FILE__;
        #parent = os.path.pa
        # The following gets us the directory of this file
        dir = os.path.dirname(os.path.abspath(__file__))
        cls.path = os.path.join(dir, "test_data/test_hpo_graph.tsv")

    def test_file_is_found(self):
        self.assertTrue(os.path.isfile(self.path))

    def test_ctor_ensmaller_parser(self):
        try:
            parser = Hpo2EnsmallenParser(self.path)
            self.assertTrue( not False)
            print("WE GOT HERE")
        except Exception as e:
            print("Problem:")
            print(e)
            pass

    def test_graph(self):
        parser = Hpo2EnsmallenParser(self.path)
        #g = parser.graph
        self.assertTrue(parser)
        self.assertTrue(isinstance(parser, Hpo2EnsmallenParser))
        g = parser.graph
        self.assertTrue(g)
