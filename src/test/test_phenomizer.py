from unittest import TestCase
from semanticsimilarity.phenomizer import Phenomizer
from semanticsimilarity.resnik import Resnik
from semanticsimilarity.hpo_ensmallen import HpoEnsmallen
from semanticsimilarity.annotation_counter import AnnotationCounter
from pyspark.sql import SparkSession, DataFrame
from parameterized import parameterized
import os
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
from pyspark.sql import functions as F
from semanticsimilarity.phenomizer import TestPt
from collections import defaultdict


class TestPhenomizer(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark_obj = SparkSession.builder.appName("pandas to spark").getOrCreate()

        # The following gets us the directory of this file
        dir = os.path.dirname(os.path.abspath(__file__))
        cls.hpo_path = os.path.join(dir, "test_data/test_hpo_graph.tsv")
        cls.hpo_path_tiny = os.path.join(dir, "test_data/test_hpo_graph_tiny.tsv")
        cls.hpo_annotations_path = os.path.join(dir, "test_data/test_hpo_annotations.tsv")

        # make an ensmallen object for HPO
        cls.hpo_ensmallen = HpoEnsmallen(cls.hpo_path)

        # make an ensmallen object for HPO-A
        cls.hpo_a_ensmallen = HpoEnsmallen(cls.hpo_annotations_path) # This doesn't work, might be wrong format. Is this the right spot anyway? ***
        # cls.hpo_a_ensmallen = HpoEnsmallen(cls.hpo_path)

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

        # make a fake disease set to generate term counts ***Do we need a separate annoation counter for diseases?
        cls.diseaseAnnotationCounter = AnnotationCounter(hpo=cls.hpo_a_ensmallen)
        # create a very trivial list of diseases and features (subset of actual disease-phenotype annotations)
        # Abnormal nervous system physiology HP:0012638
        # Abnormality of the nervous system HP:0000707
        # Phenotypic abnormality HP:0000118
        # So adding HP:0012638 should give us one count for these three terms
        disease_annots = []
        for d in [
                  # MONDO:0019391: Fanconi Anemia
                  {'disease_id': "MONDO:0019391", 'hpo_id': 'HP:0004322'},  # Short stature
                  {'disease_id': "MONDO:0019391", 'hpo_id': 'HP:0002823'},  # Abnormality of femur morphology
                  {'disease_id': "MONDO:0019391", 'hpo_id': 'HP:0000252'},  # Microcephaly
                  {'disease_id': "MONDO:0019391", 'hpo_id': 'HP:0000175'},  # Cleft palate
                  {'disease_id': "MONDO:0019391", 'hpo_id': 'HP:0001903'},  # Anemia
                  {'disease_id': "MONDO:0019391", 'hpo_id': 'HP:0000492'},  # Abnormal eyelid morphology
                  {'disease_id': "MONDO:0019391", 'hpo_id': 'HP:0000324'},  # Facial asymmetry
                  {'disease_id': "MONDO:0019391", 'hpo_id': 'HP:0012210'},  # Abnormal renal morphology
                  {'disease_id': "MONDO:0019391", 'hpo_id': 'HP:0000083'},  # Renal insufficiency
                  {'disease_id': "MONDO:0019391", 'hpo_id': 'HP:0001873'},  # Thrombocytopenia
                  {'disease_id': "MONDO:0019391", 'hpo_id': 'HP:0002414'},  # Spina bifida
                  {'disease_id': "MONDO:0019391", 'hpo_id': 'HP:0008572'},  # External ear malformation
                  {'disease_id': "MONDO:0019391", 'hpo_id': 'HP:0001760'},  # Abnormal foot morphology
                  # MONDO:0007523: Ehlers-Danlos syndrome, hypermobility type
                  {'disease_id': "MONDO:0007523", 'hpo_id': 'HP:0000963'},  # Thin skin
                  {'disease_id': "MONDO:0007523", 'hpo_id': 'HP:0012378'},  # Fatigue
                  {'disease_id': "MONDO:0007523", 'hpo_id': 'HP:0003042'},  # Elbow dislocation
                  {'disease_id': "MONDO:0007523", 'hpo_id': 'HP:0002829'},  # Arthralgia
                  {'disease_id': "MONDO:0007523", 'hpo_id': 'HP:0002827'},  # Hip dislocation
                  {'disease_id': "MONDO:0007523", 'hpo_id': 'HP:0001760'},  # Abnormal foot morphology
                  {'disease_id': "MONDO:0007523", 'hpo_id': 'HP:0002024'},  # Malabsorption
                  {'disease_id': "MONDO:0007523", 'hpo_id': 'HP:0000974'},  # Hyperextensible skin
                  {'disease_id': "MONDO:0007523", 'hpo_id': 'HP:0002650'},  # Scoliosis
                  {'disease_id': "MONDO:0007523", 'hpo_id': 'HP:0001388'},  # Joint laxity
                  {'disease_id': "MONDO:0007523", 'hpo_id': 'HP:0011675'},  # Arrhythmia
                  {'disease_id': "MONDO:0007523", 'hpo_id': 'HP:0000023'},  # Inguinal hernia
                  {'disease_id': "MONDO:0007523", 'hpo_id': 'HP:0000563'}]:  # Keratoconus

            disease_annots.append(d)

        cls.disease_pd = pd.DataFrame(disease_annots)
        cls.disease_sdf = cls.spark_obj.createDataFrame(cls.disease_pd)
        cls.diseaseAnnotationCounter.add_counts(cls.disease_pd, patient_id_col='disease_id')

        # make Resnik object
        cls.resnik = Resnik(counts_d=cls.annotationCounter.get_counts_dict(),
                            total=13, ensmallen=cls.hpo_ensmallen)

        # make Resnik object ***Make new Resnik object for diseases?
        cls.diseaseResnik = Resnik(counts_d=cls.diseaseAnnotationCounter.get_counts_dict(),
                                   total=26, ensmallen=cls.hpo_ensmallen)  # 26 or 2? Is this total diseases, total annotations?

        # The above are for testing individual components needed to make phenomizer object
        # Below are things we are using to test make_patient_similarity_dataframe()
        # which makes an HpoEnsmallen, AnnotationCounter, Resnik, and Phenomizer object itself

        # make HPO spark df
        cls.hpo_pd = pd.read_csv(cls.hpo_path)
        cls.hpo_spark = cls.spark_obj.createDataFrame(cls.hpo_pd)

        # make HPO A spark df
        cls.hpoa_pd = pd.read_csv(cls.hpo_annotations_path)
        cls.hpoa_spark = cls.spark_obj.createDataFrame(cls.hpoa_pd)

        # make patient_df spark dataframe
        cls.patient_spark = cls.spark_obj.createDataFrame(cls.patient_pd)

        # make disease_df spark dataframe for patient x disease similarity testing
        cls.disease_spark = cls.spark_obj.createDataFrame(cls.disease_pd)

        # make three held out patients
        holdout_annots = []
        for d in [
                  {'patient_id': "100", 'hpo_id': 'HP:0000118'},
                  {'patient_id': "101", 'hpo_id': 'HP:0009124'},
                  {'patient_id': "101", 'hpo_id': 'HP:0100881'},
                  {'patient_id': "102", 'hpo_id': 'HP:0410008'},
                  {'patient_id': "200", 'hpo_id': 'HP:0009124'},
                  {'patient_id': "200", 'hpo_id': 'HP:0410008'},
                  {'patient_id': "200", 'hpo_id': 'HP:0009025'},
                  {'patient_id': "200", 'hpo_id': 'HP:0000818'},
                  {'patient_id': "200", 'hpo_id': 'HP:0100881'},
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

        # make some test to cluster similarity for average_max_similarity and max_similarity
        test_cluster_similarity_list = []
        for d in [
                  # max sim for test_patient_id 1 is 30
                  # max sim for test_patient_id 2 is 60
                  # average max sim is 45
                  {'test_patient_id': "1", 'clustered_patient_id': "3",  'cluster': '1', 'sim': 20},
                  {'test_patient_id': "1", 'clustered_patient_id': "4",  'cluster': '1', 'sim': 30},
                  {'test_patient_id': "1", 'clustered_patient_id': "5",  'cluster': '1', 'sim': 40},
                  {'test_patient_id': "1", 'clustered_patient_id': "6",  'cluster': '2', 'sim': 1},
                  {'test_patient_id': "1", 'clustered_patient_id': "7",  'cluster': '2', 'sim': 1},
                  {'test_patient_id': "1", 'clustered_patient_id': "8",  'cluster': '2', 'sim': 1},
                  {'test_patient_id': "1", 'clustered_patient_id': "9",  'cluster': '2', 'sim': 1},
                  {'test_patient_id': "2", 'clustered_patient_id': "10", 'cluster': '1', 'sim': 3},
                  {'test_patient_id': "2", 'clustered_patient_id': "11", 'cluster': '1', 'sim': 3},
                  {'test_patient_id': "2", 'clustered_patient_id': "12", 'cluster': '1', 'sim': 3},
                  {'test_patient_id': "2", 'clustered_patient_id': "13", 'cluster': '2', 'sim': 50},
                  {'test_patient_id': "2", 'clustered_patient_id': "14", 'cluster': '2', 'sim': 60},
                  {'test_patient_id': "2", 'clustered_patient_id': "15", 'cluster': '2', 'sim': 70}]:
            test_cluster_similarity_list.append(d)
        cls.test_cluster_similarity_pd = pd.DataFrame(test_cluster_similarity_list)
        cls.test_cluster_similarity = cls.spark_obj.createDataFrame(cls.test_cluster_similarity_pd)

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
        heldout_patient = self.holdout_patients.filter(F.col("patient_id") == 200)
        sim = p.patient_to_cluster_similarity(test_patient_hpo_terms=heldout_patient,
                                              clustered_patient_hpo_terms=self.patient_sdf,
                                              cluster_assignments=self.cluster_assignment)
        self.assertTrue(isinstance(sim, pd.DataFrame))
        self.assertEqual(len(sim), self.cluster_assignment.count())

    def test_patient_to_cluster_similarity_method_returns_list_with_correct_features_pd(self):
        assigned_clusters = [i[0] for i in self.cluster_assignment.select('cluster').distinct().collect()]
        p = Phenomizer(self.resnik.get_mica_d())
        heldout_patient = self.holdout_patients.filter(F.col("patient_id") == 200)
        sim = p.patient_to_cluster_similarity_pd(test_patient_hpo_terms=heldout_patient,
                                                 clustered_patient_hpo_terms=self.patient_sdf,
                                                 cluster_assignments=self.cluster_assignment)
        self.assertTrue(isinstance(sim, pd.DataFrame))
        self.assertEqual(len(sim), self.cluster_assignment.count())

    def test_generalizability(self):
        p = Phenomizer(self.resnik.get_mica_d())
        df = p.center_to_cluster_generalizability(test_patients_hpo_terms=self.holdout_patients,
                                              clustered_patient_hpo_terms=self.patient_sdf,
                                              cluster_assignments=self.cluster_assignment)
        self.assertCountEqual(df.columns, ['mean.sim', 'sd.sim','observed','zscore'])

    def test_has_make_patient_disease_similarity_long_spark_df(self):
        p = Phenomizer({})  # initialize with empty mica_d - make_patient_similarity_dataframe will populate it itself
        self.assertTrue(hasattr(p, 'make_patient_disease_similarity_long_spark_df'))

    def test_max_similarity_cluster(self):  # nb: this is NOT testing average_max_similarity (that is below)
        p = Phenomizer(self.resnik.get_mica_d())
        self.assertTrue(hasattr(p, "max_similarity_cluster"))
        ams = p.max_similarity_cluster(self.test_cluster_similarity_pd,
                                       test_pt_col_name='test_patient_id',
                                       cluster_col_name='cluster',
                                       sim_score_col_name='sim')
        self.assertEqual(ams.__class__, pd.DataFrame)
        self.assertCountEqual(ams.columns, ['test_patient_id', 'max_cluster', 'average_similarity', 'probability'])
        pt_1 = ams.loc[ams['test_patient_id'] == '1']
        pt_2 = ams.loc[ams['test_patient_id'] == '2']
        self.assertEqual(len(pt_1), 1)
        self.assertEqual(len(pt_2), 1)
        #    test_patient_id max_cluster  average_similarity   probability
        # 0               1           1                30.0          30/31
        # 1               2           2                60.0          60/63
        assert_frame_equal(ams.loc[ams['test_patient_id'] == '1'],
                           pd.DataFrame(data={'test_patient_id': '1',
                                              'max_cluster': '1',
                                              'average_similarity': 30.0,
                                              'probability': 30/31,
                                              }, index=[0]))
        assert_frame_equal(ams.loc[ams['test_patient_id'] == '2'],
                           pd.DataFrame(data={'test_patient_id': '2',
                                              'max_cluster': '2',
                                              'average_similarity': 60.0,
                                              'probability': 60/63,
                                              }, index=[1]))

    def test_average_max_similarity(self):
        p = Phenomizer(self.resnik.get_mica_d())
        self.assertTrue(hasattr(p, "average_max_similarity"))
        ams = p.average_max_similarity(self.test_cluster_similarity_pd,
                                       test_pt_col_name='test_patient_id',
                                       cluster_col_name='cluster',
                                       sim_score_col_name='sim')
        self.assertEqual(ams, ((30/31)+(60/63))/2)

    def test_get_max_sim(self):
        patient_d = defaultdict(TestPt)
        for _, row in self.test_cluster_similarity_pd.iterrows():
            test_id = row['test_patient_id']
            cluster = row['cluster']
            score = row['sim']
            if test_id not in patient_d:
                tp = TestPt(test_id)
                patient_d[test_id] = tp
            patient_d[test_id].add_score(cluster, score)
        self.assertAlmostEqual(patient_d['1'].get_max_sim(), 30/31)
        self.assertAlmostEqual(patient_d['2'].get_max_sim(), 60/63)

    def test_get_best_cluster_and_average_score(self):
        patient_d = defaultdict(TestPt)
        for _, row in self.test_cluster_similarity_pd.iterrows():
            test_id = row['test_patient_id']
            cluster = row['cluster']
            score = row['sim']
            if test_id not in patient_d:
                tp = TestPt(test_id)
                patient_d[test_id] = tp
            patient_d[test_id].add_score(cluster, score)
        self.assertEqual(patient_d['1'].get_best_cluster_and_average_score(), ['1', 30.0, 30/31])
        self.assertEqual(patient_d['2'].get_best_cluster_and_average_score(), ['2', 60.0, 60/63])

# Below are new test additions for patient-disease similarity testing

    def test_make_patient_disease_similarity_long_spark_df(self):
        p = Phenomizer({})  # initialize with empty mica_d - make_patient_similarity_dataframe will populate it itself

        sim_df = p.make_patient_disease_similarity_long_spark_df(patient_df=self.patient_spark,
                                                         disease_df=self.disease_spark,
                                                         hpo_graph_edges_df=self.hpo_spark,
                                                         hpo_annotations_df=self.hpoa_spark,
                                                         person_id_col='patient_id',
                                                         hpo_term_col='hpo_id',
                                                         disease_id_col='disease_id')
        self.assertTrue(isinstance(sim_df, DataFrame))
        self.assertEqual(sim_df.columns, ['patient', 'disease', 'similarity'])

        num_patients = len(set(list(self.patient_pd['patient_id'])))
        num_diseases = len(set(list(self.disease_pd['disease_id'])))
        expected_rows = num_patients**num_diseases  # Expected to have a similarity score for each pairwise patient x disease combination.
        self.assertEqual(sim_df.count(), expected_rows,
                         msg=f"Didn't get expected number of rows in similarity df sim_df.count() {sim_df.count()} != expected_rows {expected_rows}")