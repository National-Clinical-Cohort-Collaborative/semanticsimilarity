from unittest import TestCase
from semanticsimilarity.phenomizer import Phenomizer
from semanticsimilarity.resnik import Resnik
from semanticsimilarity.hpo_ensmallen import HpoEnsmallen
from semanticsimilarity.annotation_counter import AnnotationCounter
import os
import pandas as pd


class TestPhenomizer(TestCase):

    @classmethod
    def setUpClass(cls):
        # The following gets us the directory of this file
        dir = os.path.dirname(os.path.abspath(__file__))
        cls.hpo_path = os.path.join(dir, "test_data/test_hpo_graph.tsv")
        cls.hpo_path_tiny = os.path.join(dir, "test_data/test_hpo_graph_tiny.tsv")

        # make an ensmallen object for HPO
        cls.hpo_ensmallen = HpoEnsmallen(cls.hpo_path)

        # make a fake population to generate term counts
        cls.annotationCounter = AnnotationCounter(hpo=cls.hpo_ensmallen)
        # create a very trivial list of patients and features
        # Abnormal nervous system physiology HP:0012638
        # Abnormality of the nervous system HP:0000707
        # Phenotypic abnormality HP:0000118
        # So adding HP:0012638 should give us one count for these three terms
        annots = []
        for d in [
                  {'patient_id': "1", 'hpo_id': 'HP:0000118'},
                  {'patient_id': "2", 'hpo_id': 'HP:0000707'},
                  {'patient_id': "3", 'hpo_id': 'HP:0000818'},
                  {'patient_id': "4", 'hpo_id': 'HP:0000834'},
                  {'patient_id': "5", 'hpo_id': 'HP:0000873'},
                  {'patient_id': "6", 'hpo_id': 'HP:0003549'},
                  {'patient_id': "7", 'hpo_id': 'HP:0009025'},
                  {'patient_id': "8", 'hpo_id': 'HP:0009124'},
                  {'patient_id': "9", 'hpo_id': 'HP:0012638'},
                  {'patient_id': "10", 'hpo_id': 'HP:0012639'},
                  {'patient_id': "11", 'hpo_id': 'HP:0100568'},
                  {'patient_id': "12", 'hpo_id': 'HP:0100881'},
                  {'patient_id': "13", 'hpo_id': 'HP:0410008'}]:
            annots.append(d)
        df = pd.DataFrame(annots)
        cls.annotationCounter.add_counts(df)

        # make Resnik object
        cls.resnik = Resnik(counts_d=cls.annotationCounter.get_counts_dict(),
                            total=13, ensmallen=cls.hpo_ensmallen)

    def test_phenomizer_simple(self):
        # make two patients
        self.patientA = set(['HP:0000707'])
        self.patientB = set(['HP:0000707'])

        p = Phenomizer(self.resnik.get_mica_d())
        ss = p.similarity_score(self.patientA, self.patientB)
        self.assertTrue(isinstance(ss, (int, float)))
        self.assertAlmostEquals(ss, 1.1786549963416462)

    def test_phenomizer_non_root_against_root(self):
        # make two patients
        self.patientA = set(['HP:0001818'])
        self.patientB = set(['HP:0000707'])

        p = Phenomizer(self.resnik.get_mica_d())
        ss = p.similarity_score(self.patientA, self.patientB)
        self.assertTrue(isinstance(ss, (int, float)))
        self.assertAlmostEquals(ss, 0)

    def test_phenomizer_leaf_against_leaf(self):
        # make two patients
        self.patientA = set(['HP:0012638'])
        self.patientB = set(['HP:0012638'])

        p = Phenomizer(self.resnik.get_mica_d())
        ss = p.similarity_score(self.patientA, self.patientB)
        self.assertTrue(isinstance(ss, (int, float)))
        self.assertAlmostEquals(ss, 2.5649493574615367)

    def test_phenomizer_two_terms_per_patient(self):
        # make two patients
        self.patientA = set(['HP:0012638', 'HP:0001818'])
        self.patientB = set(['HP:0012638', 'HP:0001818'])

        p = Phenomizer(self.resnik.get_mica_d())
        ss = p.similarity_score(self.patientA, self.patientB)
        self.assertTrue(isinstance(ss, (int, float)))
        self.assertAlmostEquals(ss, 2.5649493574615367*0.5)  # this is the IC of HP:0012638 vs HP:0012638 / 2

    def test_has_update_mica_d(self):
        p = Phenomizer(self.resnik.get_mica_d())
        self.assertTrue(hasattr(p, 'update_mica_d'))
        new_dict = {'foo': 'bar'}
        p.update_mica_d(new_dict)
        self.assertEqual(p._mica_d['foo'], 'bar')

    def test_has_make_similarity_matrix_method(self):
        p = Phenomizer(self.resnik.get_mica_d())
        self.assertTrue(hasattr(p, 'make_similarity_matrix'))