from collections import defaultdict
from clustering import HpoEnsmallen
import pandas as pd
from warnings import warn

class AnnotationCounter:

    def __init__(self, hpo: HpoEnsmallen):
        self._termcounts = defaultdict(int)
        self._total_patients = 0
        if not isinstance(hpo, HpoEnsmallen):
            raise ValueError("hpo argument must be an object of type HpoEnsmallen")
        self._hpo = hpo

    def add_counts(self, counts_df):
        if isinstance(counts_df, pd.DataFrame):
            # the dataframe must have exactly two columns, the first of
            # which is called patient_id and the second is called hpo_id
            if not len(counts_df.columns):  # TODO: add check for 
                raise ValueError("DataFrame must have exactly two columns")
            if 'patient_id' not in counts_df.columns or 'hpo_id' not in counts_df.columns:
                raise ValueError("Columns must be patient_id and hpo_id, but we got {}".format(";".join(counts_df.columns)))
            # Group by patient_id and create a dataframe with one row per patient_id
            # as well as a list of the hpo_id's for that patient id
            df_by_patient_id = counts_df.groupby('patient_id')['hpo_id'].apply(list)
            print(df_by_patient_id)
            # now we can create a set that contains all of the ancestors of all terms
            # to which each patient is annotated and then we use this to increment
            # the counts dictionary.
            # df_by_patient_id is a pandas series, whence iteritems()
            for index, row in df_by_patient_id.iteritems():
                # pat_id = row[0]
                hpo_id_list = row
                ######## TODO #################
                # remove the following before production, but now
                # we still need to check sanity
                if not isinstance(hpo_id_list, list):
                    raise ValueError('hpo_id not list')
                induced_ancestor_graph = set()
                for hpo_id in hpo_id_list:
                    if self._hpo.node_exists(hpo_id):
                        induced_ancestor_graph.add(hpo_id)  # add orignal term, which does not appear in ancestors
                        ancs = self._hpo.get_ancestors(hpo_id)
                        induced_ancestor_graph.update(ancs)
                    else:
                        warn(f"Couldn't find {hpo_id} in self._hpo graph")
                self._total_patients += 1
                for hpo_id in induced_ancestor_graph:
                    self._termcounts[hpo_id] += 1
        # TODO elif pyspark: do appropriate stuff

        else:
            raise ValueError("counts_df argument must be of type pd.DataFrame (TODO or spark")

    def get_total_patient_count(self):
        return self._total_patients

    def get_counts_dict(self):
        return self._termcounts
