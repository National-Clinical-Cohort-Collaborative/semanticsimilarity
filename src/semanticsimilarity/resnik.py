from collections import defaultdict
from typing import Dict
import math
import itertools
from .term_pair import TermPair
from .hpo_ensmallen import HpoEnsmallen


class Resnik:

    def __init__(self, counts_d: Dict, total: int, ensmallen: HpoEnsmallen):
        """
        The constructor calculates the information content of each term from the term counts
        """
        self._counts_d = counts_d
        self._total_count = total
        # information content dictionary
        self._ic_d = defaultdict(float)
        self._mica_d = defaultdict(float)
        for hpo_id, count in counts_d.items():
            ic = -1 * math.log(count/total)
            self._ic_d[hpo_id] = ic
        # get list of all HPO terms
        hpo_id_list = list(counts_d.keys())
        for i in itertools.combinations_with_replacement(hpo_id_list, 2):
            self._mica_d[i] = self.calculate_mica_ic(i, ensmallen)

    def calculate_mica_ic(self, tp, ensmallen: HpoEnsmallen) -> TermPair:
        t1 = tp[0]
        t2 = tp[1]
        t1ancs = ensmallen.get_ancestors(t1)
        t2ancs = ensmallen.get_ancestors(t2)
        common_ancs = t1ancs.intersection(t2ancs)
        max_ic = 0
        for t in common_ancs:
            if self._ic_d.get(t, 0) > max_ic:
                max_ic = self._ic_d.get(t, 0)
        return max_ic

    def get_mica_d(self):
        return self._mica_d
