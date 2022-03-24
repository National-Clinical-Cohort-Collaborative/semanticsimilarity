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


class Phenomizer:

    def __init__(self, mica_d: Dict):
        self._mica_d = mica_d

    def similarity_score(self, patientA: Set[str], patientB: Set[str]) -> float:
        """

        This implements equation (2) of PMID:19800049 but with both D and Q are patients.
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

    def patient_to_cluster_similarity(self,
                                      test_patient_hpo_terms: DataFrame,
                                      clustered_patient_hpo_terms: DataFrame,
                                      cluster_assignments: DataFrame,
                                      test_patient_id_col_name: str = 'patient_id',
                                      test_patient_hpo_col_name: str = 'hpo_id',
                                      cluster_assignment_patient_col_name: str = 'patient_id',
                                      cluster_assignment_cluster_col_name: str = 'cluster',
                                      clustered_patient_id_col_name: str = 'patient_id',
                                      clustered_patient_hpo_col_name: str = 'hpo_id') -> list:

        average_sim_for_pt_to_clusters = []

        test_patient_hpo_term_list = [i[0] for i in test_patient_hpo_terms.select(test_patient_hpo_col_name).distinct().collect()]

        clusters = [i[0] for i in cluster_assignments.select(cluster_assignment_cluster_col_name).distinct().collect()]

        for k in sorted(clusters):
            sim_for_pt_to_cluster_k = []
            patients_in_this_cluster = [i[0] for i in cluster_assignments.filter(F.col(cluster_assignment_cluster_col_name) == k).select(cluster_assignment_patient_col_name).collect()]
            for p in patients_in_this_cluster:
                p_hpo_ids = [i[0] for i in clustered_patient_hpo_terms.filter(F.col(clustered_patient_id_col_name) == p).select(clustered_patient_hpo_col_name).distinct().collect()]
                ss = self.similarity_score(test_patient_hpo_term_list, p_hpo_ids)
                sim_for_pt_to_cluster_k.append(ss)
            average_sim_for_pt_to_clusters.append(np.mean(sim_for_pt_to_cluster_k))
        return average_sim_for_pt_to_clusters

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
