from clustering.hpo_ensmallen_parser import Hpo2EnsmallenParser


class HpoEnsmallen:
    """A class to keep track of things like information content and maps for doing sem sim
    """

    def __init__(self, hpo_graph):
        parser = Hpo2EnsmallenParser(hpo_graph)
        self.graph = parser.graph
        self.graph_reversed_edges = parser.graph_reversed_edges

    # get all descendents
    def get_descendents(self, hpo_term, descendents=[], start=True) -> list:
        if not start:  # avoid adding original term at the beginning
            descendents += [hpo_term]

        for neighbor in self.graph_reversed_edges.get_node_neighbours_name_by_node_name(hpo_term):
            if neighbor not in descendents:
                descendents = self.get_descendents(neighbor, descendents, start=False)

        return descendents

    # get all ancestors
    def get_ancestors(self, hpo_term, ancestors=[], start=True) -> list:
        if not start:
            ancestors += [hpo_term]

        for neighbor in self.graph.get_node_neighbours_name_by_node_name(hpo_term):
            if neighbor not in ancestors:
                ancestors = self.get_ancestors(neighbor, ancestors, start=False)

        return ancestors
