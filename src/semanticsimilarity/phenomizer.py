from typing import Dict, Set
from .term_pair import TermPair


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
                if tp not in self._mica_d:
                    print("Warning, could not find tp: {}", tp)
                score = self._mica_d.get(tp, 0)
                if score > maxscore: maxscore = score
            a_to_b_sim.append(maxscore)
        for hpoB in patientB:
            maxscore = 0
            for hpoA in patientA:
                tp = TermPair(hpoA, hpoB)
                if tp not in self._mica_d:
                    print("Warning, could not find tp: {}", tp)
                score = self._mica_d.get(tp, 0)
                if score > maxscore: maxscore = score
            b_to_a_sim.append(maxscore)
        if len(a_to_b_sim) == 0 or len(b_to_a_sim) == 0:
            return 0
        return 0.5 * sum(a_to_b_sim)/len(a_to_b_sim) + 0.5 * sum(b_to_a_sim)/len(b_to_a_sim)

    def make_similarity_matrix():
        pass

    def update_mica_d(self, new_mica_d: dict):
        self._mica_d = new_mica_d
