from unittest import TestCase
from semanticsimilarity.annotation_counter import AnnotationCounter
from semanticsimilarity.hpo_ensmallen import HpoEnsmallen
from semanticsimilarity.resnik import Resnik
import os
import math
import pandas as pd


class TestResnik(TestCase):

    @classmethod
    def setUpClass(cls):
        # The following gets us the directory of this file
        dir = os.path.dirname(os.path.abspath(__file__))
        cls.hpo_path = os.path.join(dir, "test_data/test_hpo_graph.tsv")
        cls.hpo_path_tiny = os.path.join(dir, "test_data/test_hpo_graph_tiny.tsv")

    def test_mica_of_same_term_is_term(self):
        ensmallen = HpoEnsmallen(hpo_graph=self.hpo_path)
        annotationCounter = AnnotationCounter(hpo=ensmallen)
        # create a very trivial list of patients and features
        # Abnormal nervous system physiology HP:0012638
        # Abnormality of the nervous system HP:0000707
        # Phenotypic abnormality HP:0000118
        # So adding HP:0012638 should give us one count for these three terms
        annots = []
        for d in [{'patient_id': "1", 'hpo_id': 'HP:0012638'},
                  {'patient_id': "2", 'hpo_id': 'HP:0012638'},
                  {'patient_id': "3", 'hpo_id': 'HP:0000707'},
                  {'patient_id': "4", 'hpo_id': 'HP:0000707'}]:
            annots.append(d)
        df = pd.DataFrame(annots)
        annotationCounter.add_counts(df)
        self.assertEqual(4, annotationCounter.get_total_patient_count())
        counts_d = annotationCounter.get_counts_dict()
        total = annotationCounter.get_total_patient_count()
        resnik = Resnik(counts_d=counts_d, total=total, ensmallen=ensmallen)
        mica_d = resnik.get_mica_d()
        mica_0012638 = -math.log(0.5)
        mica_0000707 = -math.log(1.0)
        EPSILON = 0.000001
        self.assertEqual(mica_0012638, mica_d.get(('HP:0012638', 'HP:0012638')), EPSILON)
        self.assertEqual(mica_0000707, mica_d.get(('HP:0000707', 'HP:0000707')), EPSILON)
