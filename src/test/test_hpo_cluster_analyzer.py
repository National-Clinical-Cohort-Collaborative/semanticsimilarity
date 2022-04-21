from unittest import TestCase
import os
from semanticsimilarity import HpoEnsmallen
from semanticsimilarity.hpo_cluster_analyzer import HpoClusterAnalyzer
import pandas as pd
from pyspark.sql import SparkSession
from parameterized import parameterized
from scipy.stats import chi2_contingency, fisher_exact
import numpy as np
from datetime import datetime
import math


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
        cls.do_fisher_result_df = cls.clusterAnalyzer.do_fisher_exact()

        #
        # do_chi_square_on_covariates stuff
        #
        len_of_fake_data = 100

        # make some fake data with significant chi-square values
        idx = list(range(1, len_of_fake_data + 1))
        cluster_col_data = [1, 2, 3, 4] * int(len_of_fake_data/4)
        boolean_cov_data = [bool(i % 2) for i in idx]  # True, False, True, ...
        boolean_0_1_cov_data = [i % 2 for i in idx]  # 0, 1, 0, 1, ...
        factor_cov_data = ['Female' if v % 2 else 'Unknown' if v % 4 else 'Male' \
                           for v in idx]  # Female, Unknown, Female, Male, [repeat]

        # other edge cases
        boolean_cov_not_signif = [True for i in idx]
        boolean_cov_low_n_data = [True if i == 1 else False for i in idx]
        boolean_cov_ignore_col = [True for i in idx]

        col_names = ['cluster', 'boolean_cov', 'boolean_0_1_cov', 'factor_cov_data', 'boolean_cov_not_signif', 'boolean_cov_low_n_data', 'boolean_cov_ignore_col']
        col_data = list(zip(cluster_col_data, boolean_cov_data, boolean_0_1_cov_data, factor_cov_data, boolean_cov_not_signif, boolean_cov_low_n_data, boolean_cov_ignore_col))
        cls.do_chi_square_on_cov_df_arg = pd.DataFrame(col_data,
                                                       columns=col_names)

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
            {'HP:0000118': 6, 'HP:0000707': 1, 'HP:0000818': 4, 'HP:0000834': 1, 'HP:0000873': 1, 'HP:0003549': 1,  # cts
             'HP:0410008': 0, 'HP:0100881': 0, 'HP:0100568': 0, 'HP:0012639': 0, 'HP:0009124': 0, 'HP:0012638': 0,
             'HP:0009025': 0}
         ],
        ["1",  # cluster 1:
            [7, 8, 9, 10, 11, 12, 13],  # pt ids
            ['HP:0009025', 'HP:0009124', 'HP:0012638', 'HP:0012639', 'HP:0100568', 'HP:0100881', 'HP:0410008'],  # terms
            {'HP:0000118': 7, 'HP:0000707': 3, 'HP:0012639': 1, 'HP:0100568': 1, 'HP:0000818': 1, 'HP:0003549': 3,  # cts
             'HP:0100881': 1, 'HP:0410008': 1, 'HP:0009025': 1, 'HP:0009124': 1, 'HP:0012638': 1,
             'HP:0000834': 0, 'HP:0000873': 0
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
        self.assertTrue(
            math.isnan(self.do_chi2_result_df[self.do_chi2_result_df['hpo_id'] == 'HP:0000118']['p'].iloc[0])
        )

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

    @parameterized.expand([
        ['HP:0000818', '1-total', 7],
        ['HP:0000818', '1-with', 1],
        ['HP:0000818', '1-without', 6],
        ['HP:0000818', '0-total', 6],
        ['HP:0000818', '0-with', 4],
        ['HP:0000818', '0-without', 2],
        #  manually calculate chi2 with counts for HP:0000818 and make sure stats are what we expect
        #  table = [[with for cluster0, with for cluster1], [without for cluster0, without for cluster1]]
        ['HP:0000818', 'oddsr', fisher_exact([[4, 1], [2, 6]])[0]],  # second ret val is p
        ['HP:0000818', 'p', fisher_exact([[4, 1], [2, 6]])[1]],  # third ret val is dof
    ])
    def test_do_fisher_exact_manually_check_one_row(self, this_hpo_id, col, value):
        this_row = self.do_fisher_result_df[self.do_fisher_result_df['hpo_id'] == this_hpo_id]
        self.assertEqual(this_row[col].values[0], value)

    def test_do_chi_square_on_covariates_exists(self):
        self.assertTrue(hasattr(HpoClusterAnalyzer, "do_chi_square_on_covariates"))

    def test_do_chi_square_on_covariates_returns_pd_dataframe(self):
        self.assertTrue(isinstance(
                                   HpoClusterAnalyzer.do_chi_square_on_covariates(covariate_dataframe=self.do_chi_square_on_cov_df_arg),
                                   pd.DataFrame))

    def test_do_chi_square_on_covariates_returns_complains_about_bad_cluster_col_arg(self):
        kwargs = {'covariate_dataframe': self.do_chi_square_on_cov_df_arg, 'cluster_col': 'some_col_that_doesnt_exist'}
        self.assertRaises(ValueError, HpoClusterAnalyzer.do_chi_square_on_covariates, **kwargs)

    @parameterized.expand([
        ['boolean_cov', 'chi2', 100, False],
        ['boolean_cov', 'p', 0, True],
        ['boolean_cov', 'dof', 3, False],
        ['boolean_cov', 'cluster1', 25, False],
        ['boolean_cov', 'cluster2', 0, False],
        ['boolean_cov', 'cluster3', 25, False],
        ['boolean_cov', 'cluster4', 0, False],

        ['boolean_0_1_cov', 'chi2', 100, False],
        ['boolean_0_1_cov', 'p', 0, True],
        ['boolean_0_1_cov', 'dof', 3, False],
        ['boolean_0_1_cov', 'cluster1', 25, False],
        ['boolean_0_1_cov', 'cluster2', 0, False],
        ['boolean_0_1_cov', 'cluster3', 25, False],
        ['boolean_0_1_cov', 'cluster4', 0, False],

        ['factor_cov_data', 'chi2', 200, False],
        ['factor_cov_data', 'p', 0, True],
        ['factor_cov_data', 'dof', 6, False],
        ['factor_cov_data', 'cluster1', float('NaN'), False],
        ['factor_cov_data', 'cluster2', float('NaN'), False],
        ['factor_cov_data', 'cluster3', float('NaN'), False],
        ['factor_cov_data', 'cluster4', float('NaN'), False],

        ['boolean_cov_not_signif', 'chi2', 0, False],
        ['boolean_cov_not_signif', 'p', 1, True],
        ['boolean_cov_not_signif', 'dof', 0, False],
        ['boolean_cov_not_signif', 'cluster1', 25, False],
        ['boolean_cov_not_signif', 'cluster2', 25, False],
        ['boolean_cov_not_signif', 'cluster3', 25, False],
        ['boolean_cov_not_signif', 'cluster4', 25, False],

        ['boolean_cov_low_n_data', 'chi2', float("NaN"), False],
        ['boolean_cov_low_n_data', 'p', float("NaN"), False],
        ['boolean_cov_low_n_data', 'dof', float("NaN"), False],
        ['boolean_cov_low_n_data', 'cluster1', 1, False],
        ['boolean_cov_low_n_data', 'cluster2', 0, False],
        ['boolean_cov_low_n_data', 'cluster3', 0, False],
        ['boolean_cov_low_n_data', 'cluster4', 0, False],
    ])
    def test_do_chi_square_on_covariates_bool(self, this_covariate, stat_name, exp_val, almost_eq):
        contingency_table = pd.crosstab(self.do_chi_square_on_cov_df_arg['cluster'],
                                        self.do_chi_square_on_cov_df_arg[this_covariate])
        chi2, p_value, dof, exp = chi2_contingency(contingency_table)
        return_pd = HpoClusterAnalyzer.do_chi_square_on_covariates(covariate_dataframe=self.do_chi_square_on_cov_df_arg)

        actual_val = return_pd.loc[return_pd['covariate'] == this_covariate][stat_name].values[0]

        if math.isnan(exp_val):
            self.assertTrue(math.isnan(actual_val))
        elif almost_eq:
            self.assertAlmostEqual(actual_val, exp_val)
        else:
            self.assertEqual(actual_val, exp_val)

    def test_do_chi_square_on_covariates_ignore_col(self):
        return_pd = HpoClusterAnalyzer.do_chi_square_on_covariates(covariate_dataframe=self.do_chi_square_on_cov_df_arg,
                                                                   ignore_col=['boolean_cov_ignore_col'])
        self.assertTrue('boolean_cov_ignore_col' not in list(return_pd['covariate'].unique()))

    def test_do_chi_square_on_covariates_bonferroni(self):
        return_pd = HpoClusterAnalyzer.do_chi_square_on_covariates(covariate_dataframe=self.do_chi_square_on_cov_df_arg,
                                                                   ignore_col=['boolean_cov_ignore_col'])
        return_pd_no_bf = HpoClusterAnalyzer.do_chi_square_on_covariates(covariate_dataframe=self.do_chi_square_on_cov_df_arg,
                                                                         ignore_col=['boolean_cov_ignore_col'],
                                                                         bonferroni=False)
        self.assertAlmostEqual(return_pd_no_bf['p'][0] * return_pd_no_bf.shape[0], return_pd['p'][0], places=21)
        self.assertTrue(max(return_pd['p'] <= 1))
        self.assertTrue(max(return_pd_no_bf['p'] <= 1))