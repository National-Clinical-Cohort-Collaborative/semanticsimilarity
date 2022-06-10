from typing import Dict, Set, Union
from warnings import warn
from .term_pair import TermPair
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.sql import SparkSession, DataFrame
from semanticsimilarity.hpo_ensmallen import HpoEnsmallen
from semanticsimilarity.resnik import Resnik
from semanticsimilarity.annotation_counter import AnnotationCounter
from collections import defaultdict


class Phenomizer:
    """Phenomizer class for measuring semantic similarity between patients using phenotype data

    """

    def __init__(self, mica_d: Dict):
        """Constructor

        :param mica_d: a dictionary containing the information content for the most
        informative common ancestor for every pair of HPO terms
        """
        self._mica_d = mica_d

    def similarity_score(self, patientA: Set[str], patientB: Set[str]) -> float:
        """This implements equation (2) of PMID:19800049 but both D and Q are patients
        (whereas in that paper, D is phenotypes for a disease, and Q is patient phenotypes)

        :param patientA: first patient phenotypes
        :param patientB: second patient phenotypes

        :return: float
        """
        a_to_b_sim = []
        b_to_a_sim = []
        for hpoA in patientA:
            maxscore = 0
            for hpoB in patientB:
                tp = TermPair(hpoA, hpoB)
                self.check_term_pair_in_mica_d(tp)
                score = self._mica_d.get((tp._t1, tp._t2), 0)
                if score > maxscore:
                    maxscore = score
            a_to_b_sim.append(maxscore)
        for hpoB in patientB:
            maxscore = 0
            for hpoA in patientA:
                tp = TermPair(hpoA, hpoB)
                self.check_term_pair_in_mica_d(tp)
                score = self._mica_d.get((tp._t1, tp._t2), 0)
                if score > maxscore:
                    maxscore = score
            b_to_a_sim.append(maxscore)
        if len(a_to_b_sim) == 0 or len(b_to_a_sim) == 0:
            return 0
        return 0.5 * sum(a_to_b_sim)/len(a_to_b_sim) + 0.5 * sum(b_to_a_sim)/len(b_to_a_sim)

    def check_term_pair_in_mica_d(self, tp: TermPair):
        """Helper method to check whether a given term pair is in self._mica_d

        :param tp: A TermPair object representing the pair of HPO terms to be checked
        :return: None
        """
        if (tp._t1, tp._t2) not in self._mica_d:
            print(f"Warning, could not find tp in self._mica_d: {(tp._t1, tp._t2)}")

    def update_mica_d(self, new_mica_d: dict):
        """Replace self._mica_d with new_mica_d

        :param new_mica_d: a dictionary containing the information content for the most
        informative common ancestor for every pair of HPO terms
        :return: None
        """
        self._mica_d = new_mica_d

    def make_patient_similarity_long_spark_df(self,
                                              patient_df,
                                              hpo_graph_edges_df,
                                              person_id_col: str = 'person_id',
                                              hpo_term_col: str = 'hpo_term'
                                              ) -> DataFrame:
        """Produce long spark dataframe with similarity between all pairs of patients in patient_df

        Take a long spark dataframe of patients (containing person_ids and hpo terms) like so:

        .. list-table::
           :widths: 25 25
           :header-rows: 1

           * - person_id
             - hpo_term
           * - patient1
             - HP:0001234
           * - patient1
             - HP:0003456
           * - patient2
             - HP:0006789
           * - patient3
             - HP:0004567

        and an HPO graph like so:

        .. list-table::
           :widths: 25 25 25
           :header-rows: 1

           * - subject
             - edge_label
             - object
           * - HP:0003456
             - subclass_of
             - HP:0003456
           * - HP:0004567
             - subclass_of
             - HP:0006789

        and output a matrix of patient patient phenotypic semantic similarity like so:

        .. list-table::
           :widths: 25 25 25
           :header-rows: 1

           * - patientA
             - patientB
             - similarity
           * - patient1
             - patient1
             - 2.566
           * - patient1
             - patient2
             - 0.523
           * - patient1
             - patient3
             - 0.039
           * - patient2
             - patient2
             - 2.934
           * - patient2
             - patient3
             - 0.349

        Phenotypic similarity between patients is calculated using similarity_score above.
        Similarity of each patient to themself if also calculated.

        HPO term frequency is calculated using the frequency of each HPO term in patient_df.
        This frequency is used in the calculation of most informative common ancestor for
        each possible pair of terms.

        Note that this method updates self._mica_d using data in patient_df.

        :param patient_df: long table (spark dataframe) with person_id hpo_id for all patients
        :param hpo_graph_edges_df: HPO graph spark dataframe (with three cols: subject subclass_of object)
        :param person_id_col: name of person ID column [person_id]
        :param hpo_term_col: name of hpo term column [hpo_id]
        :return: Spark dataframe with patientA  patientB similarity for all pairs of patients (ignoring ordering)
        """

        # make HPO graph in the correct format
        output_filename = "hpo_out.tsv"
        hpo_graph_edges_df.toPandas().to_csv(output_filename)

        # make an ensmallen object for HPO
        hpo_ensmallen = HpoEnsmallen(output_filename)

        # generate term counts
        annotationCounter = AnnotationCounter(hpo=hpo_ensmallen)

        annots = []
        for row in patient_df.rdd.toLocalIterator():
            d = {'patient_id': row[person_id_col], 'hpo_id': row[hpo_term_col]}
            annots.append(d)
        df = pd.DataFrame(annots)
        annotationCounter.add_counts(df)

        # count patients
        patient_count = patient_df.dropDuplicates([person_id_col]).count()
        print(f"we have this many patient -> hpo assertions {patient_df.count()}")
        print(f"we have this many patients {patient_count}")

        # make Resnik object
        resnik = Resnik(counts_d=annotationCounter.get_counts_dict(),
                        total=patient_count,
                        ensmallen=hpo_ensmallen)

        # group by person_id to make all HPO terms for a given patient, put in a new df person_id: set([HPO terms])
        hpo_terms_by_patient = patient_df.groupBy(person_id_col).agg(F.collect_set(col(hpo_term_col))).collect()  # noqa

        # loop through hpo_terms_by_patient and compare every two patients
        self.update_mica_d(resnik.get_mica_d())

        patient_similarity_matrix = []  # list of lists

        for i, row in enumerate(hpo_terms_by_patient):
            for j in range(i, len(hpo_terms_by_patient)):
                ss = self.similarity_score(hpo_terms_by_patient[i][1],
                                           hpo_terms_by_patient[j][1])
                patient_similarity_matrix.append(
                    [
                        hpo_terms_by_patient[i][0],
                        hpo_terms_by_patient[j][0],
                        ss
                    ]
                )

        patient_similarity_matrix_pd = pd.DataFrame(patient_similarity_matrix,
                                                    columns=['patientA', 'patientB', 'similarity'])
        spark = SparkSession.builder.appName("pandas to spark").getOrCreate()
        return spark.createDataFrame(patient_similarity_matrix_pd)

    def center_to_cluster_generalizability(self,
                                           test_patients_hpo_terms:DataFrame,
                                           clustered_patient_hpo_terms: DataFrame,
                                           cluster_assignments: DataFrame,
                                           test_patient_id_col_name: str = 'patient_id',
                                           test_patient_hpo_col_name: str = 'hpo_id',
                                           cluster_assignment_patient_col_name: str = 'patient_id',
                                           cluster_assignment_cluster_col_name: str = 'cluster',
                                           clustered_patient_id_col_name: str = 'patient_id',
                                           clustered_patient_hpo_col_name: str = 'hpo_id'):
        """Measure generalizability of clusters using data frame different hospital
        system

        :param test_patients_hpo_terms: dataframe with patient HPO terms
        :param clustered_patient_hpo_terms: dataframe with HPO terms for clustered patients
        :param cluster_assignments: dataframe with cluster assignments
        :param test_patient_id_col_name: column in test_patients_hpo_terms with patient ID [patient_id]
        :param test_patient_hpo_col_name: column in test_patients_hpo_terms with HPO term [hpo_id]
        :param cluster_assignment_patient_col_name: column in cluster assignment df with patient ID [patient_id]
        :param cluster_assignment_cluster_col_name: column in cluster assignment df with cluster ID [cluster]
        :param clustered_patient_id_col_name: column in clustered patient df with patient ID [patient_id]
        :param clustered_patient_hpo_col_name: column in clustered patient df with HPO term [hpo_id]
        :return: pandas dataframe with columns mean.sim, sd.sim, observed, zscore
        """
        # 1) Generate matrix of similarities between the new ('test') patients and the existing ('clustered_patient_hpo_terms') patients
        # data frame with the columns test_pat_id, clustered_pat_id, cluster, similarity_score
        # if there are M test patients and N clustered patients, then we have MN rows and 4 columns
        test_to_clustered_df = self.patient_to_cluster_similarity_pd(test_patients_hpo_terms,
                                                                  clustered_patient_hpo_terms,
                                                                  cluster_assignments,
                                                                  test_patient_id_col_name,
                                                                  test_patient_hpo_col_name,
                                                                  cluster_assignment_patient_col_name,
                                                                  cluster_assignment_cluster_col_name,
                                                                  clustered_patient_id_col_name,
                                                                  clustered_patient_hpo_col_name)
        # 2) Calculate the OBSERVED probabilities of a test patient to belong one of the existing clusters
        # -- refactor patient_to_cluster_similarity to start with the new matrix
        original_cluster_assignments = test_to_clustered_df['cluster'].to_numpy()
        observed_max_sim = self.average_max_similarity(test_to_clustered_df)
        # 3) Do W=1000 permutations
        # -- shuffled the column with the assigned cluster and repeat
        W = 1000
        permuted_max_sim = []
        for w in range(W):
            np.random.shuffle(original_cluster_assignments)
            test_to_clustered_df['cluster'] = original_cluster_assignments
            max_sim = self.average_max_similarity(test_to_clustered_df)
            permuted_max_sim.append(max_sim)
        ## todo, calculate mean, sd, z score, number of times permuted data is higher than observed_max_sim
        mean_sim = np.mean(permuted_max_sim)
        sd_sim = np.std(permuted_max_sim)
        zscore = (observed_max_sim - mean_sim)/sd_sim
        d = {"mean.sim": mean_sim, "sd.sim": sd_sim, "observed": observed_max_sim, 'zscore':zscore}
        return pd.DataFrame([d])

    def average_max_similarity(self, test_to_clustered_df: pd.DataFrame):
        """Helper method to measure average similarity

        :param test_to_clustered_df: pandas dataframe
        :return: average sim
        """
        # group by test patient id
        # d = {'test.id':p, 'clustered.id': clustered_pat_id, 'cluster': k, 'score': ss}
        class TestPt:
            def __init__(self, id):
                self.id = id
                self.cluster_d = defaultdict(list)

            def add_score(self, cluster, score):
                self.cluster_d[cluster].append(score)

            def get_max_sim(self):
                patient_scores = []
                for cluster, scores in self.cluster_d.items():
                    mean_score = np.mean(scores)
                    patient_scores.append(mean_score)
                total_scores = np.sum(patient_scores)
                if total_scores == 0:
                    return 0
                max_score = np.max(patient_scores)/total_scores
                return max_score

        patient_d = defaultdict(TestPt)
        for _, row in test_to_clustered_df.iterrows():
            test_id = row['test.pt.id']
            cluster = row['cluster']
            score = row['score']
            if test_id not in patient_d:
                tp = TestPt(test_id)
                patient_d[test_id] = tp
            patient_d[test_id].add_score(cluster, score)
        max_sim = []
        for pat_id, testPt in patient_d.items():
            max_sim.append(testPt.get_max_sim())
        return np.mean(max_sim)

    def patient_to_cluster_similarity(self,
                                      test_patient_hpo_terms: DataFrame,
                                      clustered_patient_hpo_terms: DataFrame,
                                      cluster_assignments: DataFrame,
                                      test_patient_id_col_name: str = 'patient_id',
                                      test_patient_hpo_col_name: str = 'hpo_id',
                                      cluster_assignment_patient_col_name: str = 'patient_id',
                                      cluster_assignment_cluster_col_name: str = 'cluster',
                                      clustered_patient_id_col_name: str = 'patient_id',
                                      clustered_patient_hpo_col_name: str = 'hpo_id') -> pd.DataFrame:
        """Measure similarity of patient to cluster
        (note that we are currently using patient_to_cluster_similarity_pd instead of
        this method)

        The purpose of this method is to make a dataframe with the following columns:
        test_pat_id (patients from the 'new' center)
        clustered_pat_id (patients from the center[s] in which we generated the clusters)
        cluster (the unpermuted cluster assignments of the original clustering
        similarity_score (the similarity by Phenomizer of the test patient and clustered patient in the current row)

        If there are M test patients and N clustered patients, then we have MN rows and 4 columns
        Note that we have k clusters. Each of the rows has one of the clusters

        :param test_patient_hpo_terms:
        :param clustered_patient_hpo_terms:
        :param cluster_assignments:
        :param test_patient_id_col_name:
        :param test_patient_hpo_col_name:
        :param cluster_assignment_patient_col_name:
        :param cluster_assignment_cluster_col_name:
        :param clustered_patient_id_col_name:
        :param clustered_patient_hpo_col_name:
        :return: Pandas dataframe
        """

        test_patient_ids = [i[0] for i in test_patient_hpo_terms.select(test_patient_id_col_name).distinct().collect()]
        clusters = [i[0] for i in cluster_assignments.select(cluster_assignment_cluster_col_name).distinct().collect()]

        sim_items = []

        for this_test_pt in test_patient_ids:
            test_patient_hpo_term_list = [i[0] for i in test_patient_hpo_terms.filter(F.col(test_patient_id_col_name) == this_test_pt).select(test_patient_hpo_col_name).distinct().collect()]
            for k in sorted(clusters):
                patients_in_this_cluster = [i[0] for i in cluster_assignments.filter(F.col(cluster_assignment_cluster_col_name) == k).select(cluster_assignment_patient_col_name).collect()]
                for p in patients_in_this_cluster:
                    p_hpo_ids = [i[0] for i in clustered_patient_hpo_terms.filter(F.col(clustered_patient_id_col_name) == p).select(clustered_patient_hpo_col_name).distinct().collect()]
                    ss = self.similarity_score(test_patient_hpo_term_list, p_hpo_ids)
                    d = {'test.pt.id': this_test_pt, 'clustered.pt.id': p, 'cluster': k, 'score': ss}
                    sim_items.append(d)
        return pd.DataFrame(sim_items)

    def patient_to_cluster_similarity_pd(self,
                                      test_patient_hpo_terms: DataFrame,
                                      clustered_patient_hpo_terms: DataFrame,
                                      cluster_assignments: DataFrame,
                                      test_patient_id_col_name: str = 'patient_id',
                                      test_patient_hpo_col_name: str = 'hpo_id',
                                      cluster_assignment_patient_col_name: str = 'patient_id',
                                      cluster_assignment_cluster_col_name: str = 'cluster',
                                      clustered_patient_id_col_name: str = 'patient_id',
                                      clustered_patient_hpo_col_name: str = 'hpo_id') -> pd.DataFrame:
        """Measure similarity of patient to cluster using pandas
        (reimplementation of patient_to_cluster_similarity above)

        The purpose of this method is to make a dataframe with the following columns
        test_pat_id (patients from the 'new' center)
        clustered_pat_id (patients from the center[s] in which we generated the clusters)
        cluster (the unpermuted cluster assignments of the original clustering
        similarity_score (the similarity by Phenomizer of the test patient and clustered patient in the current row)
        if there are M test patients and N clustered patients, then we have MN rows and 4 columns
        Note that we have k clusters. Each of the rows has one of the clusters

        :param test_patient_hpo_terms:
        :param clustered_patient_hpo_terms:
        :param cluster_assignments:
        :param test_patient_id_col_name:
        :param test_patient_hpo_col_name:
        :param cluster_assignment_patient_col_name:
        :param cluster_assignment_cluster_col_name:
        :param clustered_patient_id_col_name:
        :param clustered_patient_hpo_col_name:
        :return:
        """
        # 1 Make dictionaries with key=patient_id, value=set of HPO terms for both clustered and test patients
        clustered_pt_d = defaultdict(set)
        test_pt_d = defaultdict(set)
        # 1a, cluster patients
        clustered_patient_df = clustered_patient_hpo_terms.toPandas()
        for idx, row in clustered_patient_df.iterrows():
            patient_id = row[clustered_patient_id_col_name]
            hpo_id = row[clustered_patient_hpo_col_name]
            clustered_pt_d[patient_id].add(hpo_id)
        # 1b, test patients
        test_patient_df = test_patient_hpo_terms.toPandas()
        for idx, row in test_patient_df.iterrows():
            patient_id = row[test_patient_id_col_name]
            hpo_id = row[test_patient_hpo_col_name]
            test_pt_d[patient_id].add(hpo_id)
        cluster_assignments_d = defaultdict(int)
        cluster_assignments_df = cluster_assignments.toPandas()
        for idx, row in cluster_assignments_df.iterrows():
            c = row[cluster_assignment_cluster_col_name]
            p_id = row[cluster_assignment_patient_col_name]
            cluster_assignments_d[p_id] = c
        sim_items = []
        for this_test_pt, test_pt_hpo_set in test_pt_d.items():
            for this_cluster_pt, cluster_pt_hpo_set in clustered_pt_d.items():
                ss = self.similarity_score(test_pt_hpo_set, cluster_pt_hpo_set)
                #k = clusters[clusters['patient_id']==this_cluster_pt]
                k = cluster_assignments_d.get(this_cluster_pt)
                d = {'test.pt.id': this_test_pt, 'clustered.pt.id': this_cluster_pt, 'cluster': k, 'score': ss}
                sim_items.append(d)
        return pd.DataFrame(sim_items)

    @staticmethod
    def make_similarity_matrix(patient_df: DataFrame) -> Dict[str, Union[np.ndarray, list]]:
        """Make a similarity matrix from long patient dataframe

        :param patient_df: dataframe with patient-patient similarity
        :return: A dict with 'np' with a numpy array with patient-patient similarity and
        'patient_list' that defines the ordering of the rows and columns in 'np' with
        respect to patient IDs
        """

        # construct an index map for patients
        patientA_set = set([x["patientA"] for x in patient_df.select("patientA").distinct().collect()])  # noqa
        patientB_set = set([x["patientB"] for x in patient_df.select("patientB").distinct().collect()])  # noqa

        if len(patientA_set) != len(patientB_set):
            warn(f"len(patientA){len(patientA_set)} != len(patientB){len(patientB_set)}")

        unique_patients = list(patientA_set.union(patientB_set))  # noqa
        n_unique_patients = len(unique_patients)

        patient_id_map = {}
        id_counter = 0

        for p in unique_patients:
            if p not in patient_id_map:  # technically, this check is not needed since a set will have no duplicates.
                patient_id_map[p] = id_counter
                id_counter += 1

        # construct an adjaceny matrix (n X n) where n = # unique patients
        X = np.zeros((n_unique_patients, n_unique_patients))
        source_df_sim_iter = patient_df.rdd.toLocalIterator()
        for row in source_df_sim_iter:
            i = patient_id_map[row["patientA"]]
            j = patient_id_map[row["patientB"]]
            val = row["similarity"]
            X[i][j] = val
            X[j][i] = val

        return {'np': X, 'patient_list': unique_patients}
