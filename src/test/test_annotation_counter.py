from unittest import TestCase, SkipTest
from semanticsimilarity.annotation_counter import AnnotationCounter
from semanticsimilarity.hpo_ensmallen import HpoEnsmallen
import os
import pandas as pd
from pyspark.sql import SparkSession


class TestAnnotationCounter(TestCase):

    @classmethod
    def setUpClass(cls):
        # The following gets us the directory of this file
        dir = os.path.dirname(os.path.abspath(__file__))
        cls.hpo_path = os.path.join(dir, "test_data/test_hpo_graph.tsv")

        cls.ensmallen = HpoEnsmallen(hpo_graph=cls.hpo_path)

        # create a very trivial list of patients and features
        # Abnormal nervous system physiology HP:0012638
        # Abnormality of the nervous system HP:0000707
        # Phenotypic abnormality HP:0000118
        # So adding HP:0012638 should give us one count for these three terms
        annots = []
        d = {'patient_id': "1", 'hpo_id': 'HP:0012638'}
        d1 = {'patient_id': "2", 'hpo_id': 'HP:0000118'}
        annots.append(d)
        annots.append(d1)
        # see explanations below for the first term. The second term is at the root.
        # we expect the second term will have 2 annotations, and the others 1
        cls.df = pd.DataFrame(annots)

        spark = SparkSession.getActiveSession()
        cls.sdf = spark.createDataFrame(cls.df)

    def setUp(self):
        self.ac = AnnotationCounter(hpo=self.ensmallen)

    def test_trivial_case(self):
        annotationCounter = AnnotationCounter(hpo=self.ensmallen)
        annotationCounter.add_counts(self.df)
        self.assertEqual(2, annotationCounter.get_total_patient_count())
        count_d = annotationCounter.get_counts_dict()
        self.assertEqual(1, count_d.get('HP:0012638'))
        self.assertEqual(1, count_d.get('HP:0000707'))
        self.assertEqual(2, count_d.get('HP:0000118'))
        # we should get zero for other terms
        # if there is no entry for these terms, refer 0
        self.assertEqual(0, count_d.get('HP:0003549', 0))
        self.assertEqual(0, count_d.get('HP:0100881', 0))
        self.assertEqual(0, count_d.get('HP:0009124', 0))

    def test_two_patients_with_annots_pandas(self):
        self.ac.add_counts(self.df)
        self.assertEqual(2, self.ac.get_total_patient_count())
        count_d = self.ac.get_counts_dict()
        self.assertEqual(1, count_d.get('HP:0012638'))
        self.assertEqual(1, count_d.get('HP:0000707'))
        self.assertEqual(2, count_d.get('HP:0000118'))
        # we should get zero for other terms
        # if there is no entry for these terms, refer 0
        self.assertEqual(0, count_d.get('HP:0003549', 0))
        self.assertEqual(0, count_d.get('HP:0100881', 0))
        self.assertEqual(0, count_d.get('HP:0009124', 0))

    @SkipTest
    def test_two_patients_with_annots_spark(self):	
        self.ac.add_counts(self.sdf)	
        self.assertEqual(2, self.ac.get_total_patient_count())	
        count_d = self.ac.get_counts_dict()	
        self.assertEqual(1, count_d.get('HP:0012638'))	
        self.assertEqual(1, count_d.get('HP:0000707'))	
        self.assertEqual(2, count_d.get('HP:0000118'))	
        # we should get zero for other terms	
        # if there is no entry for these terms, refer 0	
        self.assertEqual(0, count_d.get('HP:0003549', 0))	
        self.assertEqual(0, count_d.get('HP:0100881', 0))	
        self.assertEqual(0, count_d.get('HP:0009124', 0))