class HpoEnsmallen:
    """A class to keep track of things like information content and maps for doing sem sim
    """

    def __init__(self, g):
        self.graph = g

    # get all descendents
    def get_descendents(self, hpo_term) -> list:
        print(dir(self.graph))
        return []

    # get all ancestors
    def get_ancestors(self, hpo_term) -> list:
        return []
