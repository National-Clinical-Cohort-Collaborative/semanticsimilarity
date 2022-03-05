from collections import defaultdict
from semanticsimilarity import HpoEnsmallen
from scipy.stats import chi2_contingency
import pyspark
from warnings import warn
import pandas as pd


class HpoClusterAnalyzer:
    """
    """

    def __init__(self, hpo: HpoEnsmallen):
        self._percluster_termcounts = defaultdict(dict)
        self._per_cluster_total_pt_count = defaultdict(int)
        self._hpo_terms = set()
        if not isinstance(hpo, HpoEnsmallen):
            raise ValueError("hpo argument must be an object of type HpoEnsmallen")
        self._hpo = hpo
        self._total_patients = 0
        self._clusters = []

    def add_counts(self,
                   patient_hpo_df,
                   cluster_assignment_df,
                   ph_patient_id_col='patient_id',
                   ph_hpo_col='hpo_id',
                   ca_patient_id_col='patient_id',
                   ca_cluster_col='cluster'):
        """
        cluster_assignment: Spark DF with two columns: patient ID and cluster assignment
        patient_hpo_d: Spark DF with two columns: patient ID and HPO term
        """
        cluster_dict = {row[ca_patient_id_col]: row[ca_cluster_col] for row in cluster_assignment_df.collect()}

        patient_ids = [row[ca_patient_id_col] for row in cluster_assignment_df.collect()]
        if len(patient_ids) != len(set(patient_ids)):
            raise ValueError(f"column {ca_patient_id_col} in cluster_assignment has duplicate rows for some patients!")

        self._unique_pt_ids = list(set([row[ca_patient_id_col] for row in cluster_assignment_df.collect()]))
        self._clusters = list(set(cluster_dict.values()))

        if not isinstance(cluster_assignment_df, pyspark.sql.dataframe.DataFrame):
            raise ValueError("cluster_assignment argument must be Spark DataFrame")
        if ca_patient_id_col not in cluster_assignment_df.columns:
            raise ValueError(f"cluster_assignment argument doesn't have {ca_patient_id_col} column")
        if ca_cluster_col not in cluster_assignment_df.columns:
            raise ValueError(f"cluster_assignment argument doesn't have {ca_cluster_col} column")
        if not isinstance(patient_hpo_df, pyspark.sql.dataframe.DataFrame):
            raise ValueError("counts_df must be a Spark DataFrame")

        # the dataframe must have exactly two columns, the first of
        # which is called patient_id and the second is called hpo_id
        if ph_patient_id_col not in patient_hpo_df.columns or ph_hpo_col not in patient_hpo_df.columns:
            raise ValueError("Columns must be patient_id and hpo_id, but we got {}".format(";".join(patient_hpo_df.columns)))

        # Group by patient_id and create a dataframe with one row per patient_id
        # as well as a list of the hpo_id's for that patient id
        df_by_patient_id = patient_hpo_df.toPandas().groupby(ph_patient_id_col)[ph_hpo_col].apply(list)
        print(df_by_patient_id)
        # now we can create a set that contains all of the ancestors of all terms
        # to which each patient is annotated and then we use this to increment
        # the counts dictionary.
        # df_by_patient_id is a pandas series, whence iteritems()
        for patient_id, row in df_by_patient_id.iteritems():
            if patient_id not in cluster_dict:
                raise ValueError(f"Could not find patient {patient_id} in cluster_assignment")
            cluster = cluster_dict.get(patient_id)
            self._per_cluster_total_pt_count[cluster] += 1
            if cluster not in self._percluster_termcounts:
                self._percluster_termcounts[cluster] = defaultdict(int)
            # pat_id = row[0]
            hpo_id_list = row
            # ####### TODO #################
            # remove the following before production, but now
            # we still need to check sanity
            if not isinstance(hpo_id_list, list):
                raise ValueError('hpo_id not list')
            induced_ancestor_graph = set()
            for hpo_id in hpo_id_list:
                self._hpo_terms.add(hpo_id)
                if self._hpo.node_exists(hpo_id):
                    induced_ancestor_graph.add(hpo_id)  # add orignal term, which does not appear in ancestors
                    ancs = self._hpo.get_ancestors(hpo_id)
                    induced_ancestor_graph.update(ancs)
                else:
                    warn(f"Couldn't find {hpo_id} in self._hpo graph")
            self._total_patients += 1
            for hpo_id in induced_ancestor_graph:
                self._percluster_termcounts[cluster][hpo_id] += 1

    def do_chi2(self):
        """
        Perform chi2 test for each term
        """
        results_list = []
        for hpo_id in self._hpo_terms:
            with_hpo_count = []
            without_hpo_count = []
            d = {'hpo_id': hpo_id}
            for cluster in self._clusters:
                cluster_with = f"{cluster}-with"
                cluster_without = f"{cluster}-without"
                cluster_total = f"{cluster}-total"
                total = self._per_cluster_total_pt_count[cluster]
                d[cluster_total] = total
                with_hpo = self._percluster_termcounts[cluster][hpo_id]
                d[cluster_with] =  with_hpo
                without_hpo = total - with_hpo
                with_hpo_count.append(with_hpo)
                without_hpo_count.append(without_hpo)
                d[cluster_without] = without_hpo
            table = [with_hpo_count, without_hpo_count]

            stat, p, dof, expected = [float('nan') for i in range(4)]
            if not any(without_hpo_count):
                warn(f"hpo_id {hpo_id} is in all clusters - can't compute chi-2")
            elif not any(with_hpo_count):
                warn(f"hpo_id {hpo_id} doesn't occur in any clusters - can't compute chi-2")
            else:
                stat, p, dof, expected = chi2_contingency(table)
            d['stat'] = stat
            d['p'] = p
            d['dof'] = dof
            d['expected'] = expected
            results_list.append(d)
        return pd.DataFrame(results_list)