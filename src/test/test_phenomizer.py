from unittest import TestCase
from semanticsimilarity.phenomizer import Phenomizer
from semanticsimilarity.resnik import Resnik
from semanticsimilarity.hpo_ensmallen import HpoEnsmallen
from semanticsimilarity.annotation_counter import AnnotationCounter
from pyspark.sql import SparkSession, DataFrame
from parameterized import parameterized
import os
import pandas as pd
import numpy as np
from pyspark.sql import functions as F


class TestPhenomizer(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark_obj = SparkSession.builder.appName("pandas to spark").getOrCreate()

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
                  {'patient_id': "1", 'hpo_id': 'HP:0000118'}, # Phenotypic abnormality
                  {'patient_id': "2", 'hpo_id': 'HP:0000707'}, #  Abnormality of the nervous system
                  {'patient_id': "2", 'hpo_id': 'HP:0000818'}, #  Abnormality of the endocrine system
                  {'patient_id': "3", 'hpo_id': 'HP:0000818'}, # Abnormality of the endocrine system
                  {'patient_id': "4", 'hpo_id': 'HP:0000834'}, #  Abnormality of the adrenal glands
                  {'patient_id': "5", 'hpo_id': 'HP:0000873'}, #  Diabetes insipidus
                  {'patient_id': "6", 'hpo_id': 'HP:0003549'}, #  Abnormality of connective tissue
                  {'patient_id': "7", 'hpo_id': 'HP:0009025'}, # Increased connective tissue 
                  {'patient_id': "8", 'hpo_id': 'HP:0009124'}, # Abnormal adipose tissue morphology 
                  {'patient_id': "9", 'hpo_id': 'HP:0012638'}, # Abnormal nervous system physiology
                  {'patient_id': "10", 'hpo_id': 'HP:0012639'}, # Abnormal nervous system morphology
                  {'patient_id': "11", 'hpo_id': 'HP:0100568'}, # Neoplasm of the endocrine system
                  {'patient_id': "12", 'hpo_id': 'HP:0100881'}, # Congenital mesoblastic nephroma (child of Abn connective!) 
                  {'patient_id': "13", 'hpo_id': 'HP:0410008'}]: # Abnormality of the peripheral nervous system 
            annots.append(d)

        cls.patient_pd = pd.DataFrame(annots)
        cls.patient_sdf = cls.spark_obj.createDataFrame(cls.patient_pd)
        cls.annotationCounter.add_counts(cls.patient_pd)

        # make Resnik object
        cls.resnik = Resnik(counts_d=cls.annotationCounter.get_counts_dict(),
                            total=13, ensmallen=cls.hpo_ensmallen)

        # The above are for testing individual components needed to make phenomizer object
        # Below are things we are using to test make_patient_similarity_dataframe()
        # which makes an HpoEnsmallen, AnnotationCounter, Resnik, and Phenomizer object itself

        # make HPO spark df
        cls.hpo_pd = pd.read_csv(cls.hpo_path)
        cls.hpo_spark = cls.spark_obj.createDataFrame(cls.hpo_pd)

        # make patient_df spark dataframe
        cls.patient_spark = cls.spark_obj.createDataFrame(cls.patient_pd)

        # make three held out patients
        holdout_annots = []
        for d in [
                  {'patient_id': "100", 'hpo_id': 'HP:0000118'},
                  {'patient_id': "101", 'hpo_id': 'HP:0009124'},
                  {'patient_id': "101", 'hpo_id': 'HP:0100881'},
                  {'patient_id': "102", 'hpo_id': 'HP:0410008'},
        ]:
            holdout_annots.append(d)
        cls.holdout_patients_pd = pd.DataFrame(holdout_annots)
        cls.holdout_patients = cls.spark_obj.createDataFrame(cls.holdout_patients_pd)

        # make some cluster assignments
        cluster_info = []
        for d in [
                  {'patient_id': "1", 'cluster': '1'},
                  {'patient_id': "2", 'cluster': '1'},
                  {'patient_id': "3", 'cluster': '1'},
                  {'patient_id': "4", 'cluster': '1'},
                  {'patient_id': "5", 'cluster': '1'},
                  {'patient_id': "6", 'cluster': '1'},
                  {'patient_id': "7", 'cluster': '1'},
                  {'patient_id': "8", 'cluster': '1'},
                  {'patient_id': "9", 'cluster': '1'},
                  {'patient_id': "10", 'cluster': '2'},
                  {'patient_id': "11", 'cluster': '2'},
                  {'patient_id': "12", 'cluster': '2'},
                  {'patient_id': "13", 'cluster': '2'}]:
            cluster_info.append(d)
        cls.cluster_assignment_pd = pd.DataFrame(cluster_info)
        cls.cluster_assignment = cls.spark_obj.createDataFrame(cls.cluster_assignment_pd)

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

    def test_has_make_patient_similarity_long_spark_df(self):
        p = Phenomizer({})  # initialize with empty mica_d - make_patient_similarity_dataframe will populate it itself
        self.assertTrue(hasattr(p, 'make_patient_similarity_long_spark_df'))

    def test_make_patient_similarity_long_spark_df(self):
        p = Phenomizer({})  # initialize with empty mica_d - make_patient_similarity_dataframe will populate it itself

        sim_df = p.make_patient_similarity_long_spark_df(patient_df=self.patient_spark,
                                                         hpo_graph_edges_df=self.hpo_spark,
                                                         person_id_col='patient_id',
                                                         hpo_term_col='hpo_id')
        self.assertTrue(isinstance(sim_df, DataFrame))
        self.assertEqual(sim_df.columns, ['patientA', 'patientB', 'similarity'])

        num_patients = len(set(list(self.patient_pd['patient_id'])))
        expected_rows = (num_patients**2/2)+num_patients/2
        self.assertEqual(sim_df.count(), expected_rows,
                         msg=f"Didn't get expected number of rows in similarity df sim_df.count() {sim_df.count()} != expected_rows {expected_rows}"
    )

    def test_contents_of_make_patient_similarity_long_spark_df(self):
        # test contents of patient_similarity_long_spark_df to make sure it's exactly what we want
        p = Phenomizer({})  # initialize with empty mica_d - make_patient_similarity_dataframe will populate it itself

        sim_df = p.make_patient_similarity_long_spark_df(patient_df=self.patient_spark,
                                                         hpo_graph_edges_df=self.hpo_spark,
                                                         person_id_col='patient_id',
                                                         hpo_term_col='hpo_id')
        sim_pd = sim_df.toPandas()

        # test that we have all patient represented in patientA and patientB
        uniq_patients_input = list(self.patient_spark.toPandas()['patient_id'].unique())
        self.assertCountEqual(uniq_patients_input, list(sim_pd['patientA'].unique()))
        self.assertCountEqual(uniq_patients_input, list(sim_pd['patientB'].unique()))

    @parameterized.expand([
        ['13', '13', 2.564949],  # test symmetry
        ['7', '8', 1.178655],
        ['4', '5', 0.955512],
        ['2', '5', 0.7166335],  # test patient with >1 term (pt2) and another with 1 term (pt5)
    ])
    def test_specific_pairs_in_patient_similarity_long_spark_df(self, patientA, patientB, sim):
        dec_places = 5

        # test contents of patient_similarity_long_spark_df to make sure it's exactly what we want
        p = Phenomizer({})  # initialize with empty mica_d - make_patient_similarity_dataframe will populate it itself

        sim_df = p.make_patient_similarity_long_spark_df(patient_df=self.patient_spark,
                                                         hpo_graph_edges_df=self.hpo_spark,
                                                         person_id_col='patient_id',
                                                         hpo_term_col='hpo_id')
        sim_pd = sim_df.toPandas()

        self.assertAlmostEqual(sim,
                               sim_pd.loc[(sim_pd['patientA'] == patientA) & (sim_pd['patientB'] == patientB)].iloc[0][2],
                               dec_places)

    def test_make_similarity_matrix(self):
        self.assertTrue(hasattr(Phenomizer, 'make_similarity_matrix'))

    def test_make_similarity_matrix_returns_dict_with_np_and_list(self):
        p = Phenomizer({})  # initialize with empty mica_d - make_patient_similarity_dataframe will populate it itself
        sim_long_df = p.make_patient_similarity_long_spark_df(patient_df=self.patient_spark,
                                                              hpo_graph_edges_df=self.hpo_spark,
                                                              person_id_col='patient_id',
                                                              hpo_term_col='hpo_id')
        sim_matrix = Phenomizer.make_similarity_matrix(sim_long_df)
        self.assertTrue(isinstance(sim_matrix, dict))
        self.assertTrue('np' in sim_matrix)
        self.assertTrue('patient_list' in sim_matrix)

        self.assertTrue(isinstance(sim_matrix['np'], np.ndarray))
        self.assertTrue(isinstance(sim_matrix['patient_list'], list))

        # test that 'np' matrix has the dimensions that comport with 'patient_list'
        uniq_pt_ct = len(sim_matrix['patient_list'])
        self.assertEqual(sim_matrix['np'].shape, (uniq_pt_ct, uniq_pt_ct))

        # test that 'patient_list' has exactly the correct members
        uniq_pt_set = set(list(sim_long_df.toPandas()['patientA'].unique()) + list(sim_long_df.toPandas()['patientB'].unique()))
        self.assertEqual(set(sim_matrix['patient_list']), uniq_pt_set)

    def test_long_df_matches_similarity_matrix(self):
        dec_places = 5

        p = Phenomizer({})
        sim_long_df = p.make_patient_similarity_long_spark_df(patient_df=self.patient_spark,
                                                              hpo_graph_edges_df=self.hpo_spark,
                                                              person_id_col='patient_id',
                                                              hpo_term_col='hpo_id')
        # TODO: chnage make_similarity_matrix to be more descriptive:
        # make_similarity_matrix_dict or some suchtiter
        sim_matrix = Phenomizer.make_similarity_matrix(sim_long_df)

        for row in sim_long_df.rdd.toLocalIterator():
            i = sim_matrix['patient_list'].index(row['patientA'])
            j = sim_matrix['patient_list'].index(row['patientB'])
            self.assertAlmostEqual(sim_matrix['np'][i][j], row['similarity'], dec_places,
                                   msg=f"sim matrix doesn't match long spark df for patient A {row['patientA']} vs patient B {row['patientB']}")
            self.assertAlmostEqual(sim_matrix['np'][j][i], row['similarity'], dec_places,
                                   msg=f"sim matrix doesn't match long spark df for patient B {row['patientB']} vs patient A {row['patientA']}")

    def test_patient_to_cluster_similarity_method_exists(self):
        p = Phenomizer({})
        self.assertTrue(hasattr(p, "patient_to_cluster_similarity"))

    def test_patient_to_cluster_similarity_method_returns_list_with_correct_features(self):
        assigned_clusters = [i[0] for i in self.cluster_assignment.select('cluster').distinct().collect()]
        p = Phenomizer(self.resnik.get_mica_d())
        heldout_patient = self.holdout_patients.filter(F.col("patient_id") == 101)
        sim = p.patient_to_cluster_similarity(test_patient_hpo_terms=heldout_patient,
                                              clustered_patient_hpo_terms=self.patient_sdf,
                                              cluster_assignments=self.cluster_assignment)
        self.assertTrue(isinstance(sim, DataFrame))
        self.assertEqual(len(sim), len(assigned_clusters))

    def test_generalizability(self):
        # test_patients_hpo_terms:DataFrame,
        #                              clustered_patient_hpo_terms: DataFrame,
        #                              cluster_assignments: DataFrame,
        patient_df = self.patient_df
        p = Phenomizer(self.resnik.get_mica_d())
