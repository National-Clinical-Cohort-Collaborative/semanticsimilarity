from unittest import TestCase
import os
from semanticsimilarity import HpoEnsmallen
from semanticsimilarity.hpo_cluster_analyzer import HpoClusterAnalyzer
import pandas as pd
from pyspark.sql import SparkSession
from parameterized import parameterized
from scipy.stats import chi2_contingency


class TestHpoClusterAnalyzer(TestCase):

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
        cls.clusterAnalyzer = HpoClusterAnalyzer(hpo=cls.hpo_ensmallen)
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
        # cluster assignments will be spark df's, so we'll follow that here
        cls.patient_df = cls.spark_obj.createDataFrame(cls.patient_pd)

        cluster_assignments = []
        for ca in [
                  {'patient_id': "1", 'cluster': '0'},
                  {'patient_id': "2", 'cluster': '0'},
                  {'patient_id': "3", 'cluster': '0'},
                  {'patient_id': "4", 'cluster': '0'},
                  {'patient_id': "5", 'cluster': '0'},
                  {'patient_id': "6", 'cluster': '0'},
                  {'patient_id': "7", 'cluster': '1'},
                  {'patient_id': "8", 'cluster': '1'},
                  {'patient_id': "9", 'cluster': '1'},
                  {'patient_id': "10", 'cluster': '1'},
                  {'patient_id': "11", 'cluster': '1'},
                  {'patient_id': "12", 'cluster': '1'},
                  {'patient_id': "13", 'cluster': '1'}]:
            cluster_assignments.append(ca)
        cls.cluster_assignment_pd = pd.DataFrame(cluster_assignments)
        cls.cluster_assignment_df = cls.spark_obj.createDataFrame(cls.cluster_assignment_pd)
        cls.clusterAnalyzer.add_counts(cls.patient_df, cls.cluster_assignment_df)
        cls.do_chi2_result_df = cls.clusterAnalyzer.do_chi2()


    def test_add_counts_total_patients_attr(self):
        self.assertEqual(self.clusterAnalyzer._total_patients, 13)

    def test_add_counts_cluster_attr(self):
        self.assertCountEqual(self.clusterAnalyzer._clusters, ['0', '1'])

    def test_add_counts_unique_pt_ids_attr(self):
        self.assertCountEqual(self.clusterAnalyzer._unique_pt_ids,
                              [str(i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]])

    def test_add_counts_isinstance(self):
        self.assertTrue(isinstance(self.clusterAnalyzer._hpo, HpoEnsmallen))

    #                                            HP:0000118 Phenotypic abnormality
    #                                     /                      \                                \
    #                                    /                         \                                \
    #     HP:0000707 Abn of the nervous sys                HP:0000818  Abn of the endocrine sys    HP:0003549 Abn of connective tissue
    #                                 /                              \                                \
    #   HP:0012638 Abnormal nervous system physiology     HP:0000834 Abn of the adrenal glands      HP:0009124 Abn adipose tissue morphology
    #   HP:0012639 Abnormal nervous system morphology     HP:0000873 Diabetes insipidus               HP:0100881 Congenital mesoblastic nephroma
    #   HP:0410008 Abn of peripheral nervous sys          HP:0100568 Neoplasm of the endocrine system   HP:0009025 Increased connective tissue
    #                                                        ^^
    #                                                        |HP:0011793 Neoplasm by anat site -> HP:0002664 Neoplasm -> HP:0000118
    @parameterized.expand([
        ["0",  # cluster 0:
            [1, 2, 3, 4, 5, 6],  # pt ids
            ['HP:0000118', 'HP:0000707', 'HP:0000818', 'HP:0000818', 'HP:0000834', 'HP:0000873', 'HP:0003549'],  # terms
            {'HP:0000118': 6, 'HP:0000707': 1, 'HP:0000818': 4, 'HP:0000834': 1, 'HP:0000873': 1, 'HP:0003549': 1}  # cts
         ],
        ["1",  # cluster 1:
            [7, 8, 9, 10, 11, 12, 13],  # pt ids
            ['HP:0009025', 'HP:0009124', 'HP:0012638', 'HP:0012639', 'HP:0100568', 'HP:0100881', 'HP:0410008'],  # terms
            {'HP:0000118': 7, 'HP:0000707': 3, 'HP:0012639': 1, 'HP:0100568': 1, 'HP:0000818': 1, 'HP:0003549': 3,  # cts
             'HP:0100881': 1, 'HP:0410008': 1, 'HP:0009025': 1, 'HP:0009124': 1, 'HP:0012638': 1,
             # TODO: For 'HP:0100568', to test HPO term with two paths to 118 Phen Abn, we maybe should add:
             # HP:0011793 Neoplasm by anat site and HP:0002664 Neoplasm - this is another path to 118 Phen Abn
             #  'HP:0011793': 1,  # Neoplasm by anat site
             #  'HP:0002664': 1   # Neoplasm
             }]
        ])
    def test_add_counts_counts_actual_hpo_terms(self, cluster, pt_ids, hpo_terms, term_counts):
        self.maxDiff = 700  # actually print out diff if test fails
        self.assertTrue(all(item in self.clusterAnalyzer._hpo_terms for item in hpo_terms))  # all terms should be in _hpo_terms
        self.assertEqual(self.clusterAnalyzer._per_cluster_total_pt_count.get(cluster), len(pt_ids))
        self.assertDictEqual(self.clusterAnalyzer._percluster_termcounts[cluster], term_counts)

    def test_do_chi2_is_pd_df(self):
        self.assertTrue(isinstance(self.do_chi2_result_df, pd.DataFrame))

    def test_do_chi2_columns(self):
        self.assertCountEqual(self.do_chi2_result_df.columns,
                              ['hpo_id', '1-total', '1-with', '1-without', '0-total', '0-with', '0-without',
                               'stat', 'p', 'dof', 'expected'])

    def test_do_chi2_has_data_for_all_hpo_terms(self):
        self.assertCountEqual(self.do_chi2_result_df['hpo_id'], list(self.clusterAnalyzer._hpo_terms))

    def test_do_chi2_phen_abn_p_value_should_be_nan(self):
        self.assertEqual(self.do_chi2_result_df[self.do_chi2_result_df['hpo_id'] == 'HP:0000118']['p'], float('nan'))

    @parameterized.expand([
        ['HP:0000818', '1-total', 7],
        ['HP:0000818', '1-with', 1],
        ['HP:0000818', '1-without', 6],
        ['HP:0000818', '0-total', 6],
        ['HP:0000818', '0-with', 4],
        ['HP:0000818', '0-without', 2],
        #  manually calculate chi2 with counts for HP:0000818 and make sure stats are what we expect
        #  table = [[with for cluster0, with for cluster1], [without for cluster0, without for cluster1]]
        ['HP:0000818', 'stat', chi2_contingency([[4, 1], [2, 6]])[0]],  # first ret val is stat
        ['HP:0000818', 'p', chi2_contingency([[4, 1], [2, 6]])[1]],  # second ret val is p
        ['HP:0000818', 'dof', chi2_contingency([[4, 1], [2, 6]])[2]],  # third ret val is dof
        ['HP:0000818', 'dof', 1],
    ])
    def test_do_chi2_manually_check_one_row(self, this_hpo_id, col, value):
        this_row = self.do_chi2_result_df[self.do_chi2_result_df['hpo_id'] == this_hpo_id]
        self.assertEqual(this_row[col].values[0], value)
