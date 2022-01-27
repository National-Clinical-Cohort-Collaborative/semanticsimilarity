class HpoEnsmallen:
    """A class to keep track of things like information content and maps for doing sem sim
    """

    def __init__(self, g):
        self.graph = g

    # get all descendents
    def get_descendents(self, hpo_term, descendents=[]) -> list:

        descendents += [hpo_term]

        for neighbor in self.graph_reversed_edges.get_node_neighbours_name_by_node_name(hpo_term):
            if neighbor not in descendents:
                descendents = self.get_descendents(neighbor, descendents)

        return descendents

    # get all ancestors
    def get_ancestors(self, hpo_term, ancestors=[]) -> list:
        ancestors += [hpo_term]

        for neighbor in self.graph.get_node_neighbours_name_by_node_name(hpo_term):
            if neighbor not in ancestors:
                ancestors = self.get_descendents(neighbor, ancestors)

        return ancestors
