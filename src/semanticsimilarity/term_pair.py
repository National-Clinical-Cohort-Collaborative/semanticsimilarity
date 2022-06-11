class TermPair():
    """A class to represent a pair of HPO terms
    """

    def __init__(self, termA: str, termB: str):
        """Constructor

        :param termA: first HPO term
        :param termB: second HPO term
        """
        thisPair = [termA, termB]
        thisPair.sort()
        self._t1 = thisPair[0]
        self._t2 = thisPair[1]

    def __eq__(self, other):
        return other and self._t1 == other._t1 and self._t2 == other._t2

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self._t1, self._t2))

    @property
    def t1(self):
        """first HPO term

        :return:
        """
        return self._t1

    @property
    def t2(self):
        """second HPO term

        :return:
        """
        return self._t2
