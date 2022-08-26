from typing import Dict, Set, Union, Optional
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


class TestPt:
    """Helper class for average_max_similarity and max_similarity
    """

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

    def get_best_cluster_and_average_score(self) -> [Optional[str], float]:
        best_cluster = None
        best_average_score = -1000
        patient_scores = []
        for cluster, scores in self.cluster_d.items():
            mean_score = np.mean(scores)
            patient_scores.append(mean_score)
            if mean_score > best_average_score:
                best_cluster = cluster
                best_average_score = mean_score
        total_scores = np.sum(patient_scores)
        return [best_cluster, best_average_score, best_average_score/total_scores]


class Phenomizer:
    """Phenomizer class for measuring semantic similarity between patients using phenotype data

    """

    def __init__(self, mica_d: Dict):
        self._mica_d = mica_d

    def similarity_score(self, patientA: Set[str], patientB: Set[str]) -> float:
        """This implements equation (2) of PMID:19800049 but with both D and Q are patients.

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
        if (tp._t1, tp._t2) not in self._mica_d:
            print(f"Warning, could not find tp in self._mica_d: {(tp._t1, tp._t2)}")

    def update_mica_d(self, new_mica_d: dict):
        self._mica_d = new_mica_d

    def make_patient_similarity_long_spark_df(self,
                                              patient_df,
                                              hpo_graph_edges_df,
                                              person_id_col: str = 'person_id',
                                              hpo_term_col: str = 'hpo_term'
                                              ) -> DataFrame:
        """Produce long spark dataframe with similarity between all pairs of patients in patient_df

        Args:
            patient_df: long table (spark dataframe) with person_id hpo_id for all patients
            hpo_graph_edges_df: HPO graph spark dataframe (with three cols: subject subclass_of object)
            person_id_col: name of person ID column [person_id]
            hpo_term_col: name of hpo term column [hpo_id]

        Returns:
            Spark dataframe with patientA  patientB similarity for all pairs of patients (ignoring ordering)

        Details:

        Take a long spark dataframe of patients (containing person_ids and hpo terms) like so:
        person_id  hpo_term
        patient1   HP:0001234
        patient1   HP:0003456
        patient2   HP:0006789
        patient3   HP:0004567
        ...

        and an HPO graph like so:
        subject     edge_label   object
        HP:0003456  subclass_of  HP:0003456
        HP:0004567  subclass_of  HP:0006789
        ...

        and output a matrix of patient patient phenotypic semantic similarity like so:
        patientA    patientB    similarity
        patient1    patient1    2.566
        patient1    patient2    0.523
        patient1    patient3    0.039
        patient2    patient2    2.934
        patient2    patient3    0.349
        ...

        Phenotypic similarity between patients is calculated using similarity_score above.
        Similarity of each patient to themself if also calculated.

        HPO term frequency is calculated using the frequency of each HPO term in patient_df.
        This frequency is used in the calculation of most informative common ancestor for
        each possible pair of terms.

        Note that this method updates self._mica_d using data in patient_df
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

    def make_patient_disease_similarity_long_spark_df(self,
                                                      patient_df,
                                                      disease_df,
                                                      hpo_graph_edges_df,
                                                      annotations_df,
                                                      person_id_col: str = 'person_id',
                                                      person_hpo_term_col: str = 'hpo_id',
                                                      disease_id_col: str = 'disease_id',
                                                      disease_hpo_term_col: str = 'hpo_id',
                                                      annot_subject_col: str = 'subject',
                                                      annot_object_col: str = 'object'
                                                      ) -> DataFrame:
        """Produce long spark dataframe with similarity between all patients in patient_df and diseases in disease_df

        Args:
            patient_df: long table (spark dataframe) with person_id hpo_id for all patients
            disease_df: long table  (spark dataframe) with disease_id hpo_id for all diseases being compared with patients
            hpo_graph_edges_df: HPO graph spark dataframe (with three cols: subject subclass_of object)
            annotations_df: long table (spark dataframe) containing the full HPO annotations disease:hpo term file.
            person_id_col: name of person ID column [person_id]
            person_hpo_term_col: name of hpo term column in the patient_df [hpo_id]
            disease_id_col: name of disease ID column [disease_id]
            disease_hpo_term_col: name of hpo term column in the disease_df [hpo_id]

        Returns:
            Spark dataframe with patient x disease similarity for all patient x disease pairings (ignoring ordering)

        Details:

        Take a long spark dataframe of patients (containing person_ids and hpo terms) like so:
        person_id  hpo_term
        patient1   HP:0001234
        patient1   HP:0003456
        patient2   HP:0006789
        patient3   HP:0004567
        ...

        Take a long spark dataframe of diseases (containing disease_ids and hpo terms) like so:
        disease_id  hpo_id
        disease1   HP:0007489
        disease1   HP:0006487
        disease2   HP:0006789
        disease3   HP:0004494
        ...

        and an HPO graph like so:
        subject     edge_label   object
        HP:0003456  subclass_of  HP:0003456
        HP:0004567  subclass_of  HP:0006789
        ...

        and an "annotations" file consisting of either the HPO annotations file
        or a patient annotations file formatted as following:
        subject         object
        OMIM:619426     HP:0001385
        OMIM:619340     HP:0001789

        or:

        subject         object
        patient1     HP:0001385
        patient2     HP:0001789
        ...

        and output a matrix of patient disease phenotypic semantic similarity like so:
        patient     disease    similarity
        patient1    disease1    2.566
        patient1    disease2    0.523
        patient1    disease3    0.039
        patient2    disease1    0.935
        patient2    disease2    2.934
        patient2    disease3    0.349
        ...

        Phenotypic similarity between patients and diseases is calculated using similarity_score above.

        HPO term frequency is calculated using the frequency of each HPO term in annotations_df.
        This frequency is used in the calculation of most informative common ancestor for
        each possible pair of terms.
        """

        # make HPO graph in the correct format
        output_filename = "hpo_out.tsv"
        hpo_graph_edges_df.toPandas().to_csv(output_filename)

        # make an ensmallen object for HPO
        hpo_ensmallen = HpoEnsmallen(output_filename)

        # generate term counts
        annotationCounter = AnnotationCounter(hpo=hpo_ensmallen)

        # count patients
        patient_count = patient_df.dropDuplicates([person_id_col]).count()
        print(f"we have this many patient -> hpo assertions {patient_df.count()}")
        print(f"we have this many patients {patient_count}")

        # count diseases
        disease_count = disease_df.dropDuplicates([disease_id_col]).count()
        print(f"we have this many disease -> hpo assertions {disease_df.count()}")
        print(f"we have this many diseases {disease_count}")

        # count annotations
        annotation_count = annotations_df.dropDuplicates([annot_subject_col]).count()
        print(f"we have this many subject -> object assertions {annotations_df.count()}")
        print(f"we have this many annotations {annotation_count}")

        # assemble annotations from annotations_df (HPO annotations or patient annotations file)
        annots = []
        for row in annotations_df.rdd.toLocalIterator():
            d = {'patient_id': row[annot_subject_col], 'hpo_id': row[annot_object_col]}
            annots.append(d)
        df = pd.DataFrame(annots)
        annotationCounter.add_counts(df)
        total_count = annotation_count

        # make Resnik object
        resnik = Resnik(counts_d=annotationCounter.get_counts_dict(),
                        total=total_count,
                        ensmallen=hpo_ensmallen)

        # group by person_id to make all HPO terms for a given patient, put in a new df person_id: set([HPO terms])
        hpo_terms_by_patient = patient_df.groupBy(person_id_col).agg(F.collect_set(col(person_hpo_term_col))).collect()  # noqa

        # group by disease_id to make all HPO terms for a given disease, put in a new df disease_id: set([HPO terms])
        hpo_terms_by_disease = disease_df.groupBy(disease_id_col).agg(F.collect_set(col(disease_hpo_term_col))).collect()  # noqa

        # Update mica_d with appropriate mica info from disease annotations.
        self.update_mica_d(resnik.get_mica_d())

        # loop through hpo_terms_by_patient and hpo_terms_by_disease and compare every patient and disease combination
        patient_disease_similarity_matrix = []  # list of lists

        for i, prow in enumerate(hpo_terms_by_patient):
            for j, drow in enumerate(hpo_terms_by_disease):
                ss = self.similarity_score(hpo_terms_by_patient[i][1],
                                           hpo_terms_by_disease[j][1])
                patient_disease_similarity_matrix.append(
                    [
                        hpo_terms_by_patient[i][0],
                        hpo_terms_by_disease[j][0],
                        ss
                    ]
                )

        patient_disease_similarity_matrix_pd = pd.DataFrame(patient_disease_similarity_matrix,
                                                            columns=['patient', 'disease', 'similarity'])
        spark = SparkSession.builder.appName("pandas to spark").getOrCreate()
        return spark.createDataFrame(patient_disease_similarity_matrix_pd)

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

    def max_similarity_cluster(self,
                               test_to_clustered_df: pd.DataFrame,
                               test_pt_col_name: str = 'test.id',
                               cluster_col_name: str = 'cluster',
                               sim_score_col_name: str = 'score') -> pd.DataFrame:
        """For each test patient, determine the cluster to which the patient has the most similarity and also the 
        average similarity of the test patient to patients in the cluster (and also a probability of the patient belonging to cluster)
        """
        for key in [test_pt_col_name, cluster_col_name, sim_score_col_name]:
            if key not in test_to_clustered_df.columns:
                raise KeyError(f"key {key} is not present in test_to_clustered_df!")

        patient_d = defaultdict(TestPt)
        for _, row in test_to_clustered_df.iterrows():
            test_id = row[test_pt_col_name]
            cluster = row[cluster_col_name]
            score = row[sim_score_col_name]
            if test_id not in patient_d:
                tp = TestPt(test_id)
                patient_d[test_id] = tp
            patient_d[test_id].add_score(cluster, score)
        pd_data = []
        for pat_id, testPt in patient_d.items():
            best_cluster, ave_score, prob = testPt.get_best_cluster_and_average_score()
            d = {'test_patient_id': pat_id, 'max_cluster': best_cluster,
                 'average_similarity': ave_score, 'probability': prob}
            pd_data.append(d)
        return pd.DataFrame(data=pd_data)

    def average_max_similarity(self,
                               test_to_clustered_df: pd.DataFrame,
                               test_pt_col_name: str = 'test.pt.id',
                               cluster_col_name: str = 'cluster',
                               sim_score_col_name: str = 'score'
                               ):
        """Calculate the average of the semantic similarity of each test patient to patients in the cluster that the 
        test patient matches best

        That is, this calculates the average similarity of all test patients to the cluster to which they are most 
        similar. This is used for testing the generalizability of the clusters using patients from "new" data partners.
        """
        for key in [test_pt_col_name, cluster_col_name, sim_score_col_name]:
            if key not in test_to_clustered_df.columns:
                raise KeyError(f"key {key} is not present in test_to_clustered_df!")

        patient_d = defaultdict(TestPt)
        for _, row in test_to_clustered_df.iterrows():
            test_id = row[test_pt_col_name]
            cluster = row[cluster_col_name]
            score = row[sim_score_col_name]
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
        """
        The purpose of this method is to make a dataframe with the following columns
        test_pat_id (patients from the 'new' center), clustered_pat_id (patients from the center[s] in which we generated
        the clusters), cluster (the unpermuted cluster assignments of the original clustering, similarity_score (the
        similarity by Phenomizer of the test patient and clustered patient in the current row)
        # if there are M test patients and N clustered patients, then we have MN rows and 4 columns
        # Note that we have k clusters. Each of the rows has one of the clusters
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
        """
        The purpose of this method is to make a dataframe with the following columns
        test_pat_id (patients from the 'new' center), clustered_pat_id (patients from the center[s] in which we generated
        the clusters), cluster (the unpermuted cluster assignments of the original clustering, similarity_score (the
        similarity by Phenomizer of the test patient and clustered patient in the current row)
        # if there are M test patients and N clustered patients, then we have MN rows and 4 columns
        # Note that we have k clusters. Each of the rows has one of the clusters
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
