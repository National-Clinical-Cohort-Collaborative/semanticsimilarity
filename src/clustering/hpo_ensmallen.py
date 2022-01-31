from clustering.hpo_ensmallen_parser import Hpo2EnsmallenParser


class HpoEnsmallen:
    """A class to keep track of things like information content and maps for doing sem sim
    """

    def __init__(self, hpo_graph):
        parser = Hpo2EnsmallenParser(hpo_graph)
        self.graph = parser.graph
        self.graph_reversed_edges = parser.graph_reversed_edges

    # get all descendents
    def get_descendents(self, hpo_term) -> list:
        stack = [hpo_term]
        path = []

        while stack:
            vertex = stack.pop()
            if vertex in path:
                continue
            path.append(vertex)
            for neighbor in self.graph_reversed_edges.get_node_neighbours_name_by_node_name(vertex):
                stack.append(neighbor)

        return path

    # get all ancestors
    def get_ancestors(self, hpo_term) -> list:
        stack = [hpo_term]
        path = []

        while stack:
            vertex = stack.pop()
            if vertex in path:
                continue
            path.append(vertex)
            for neighbor in self.graph.get_node_neighbours_name_by_node_name(vertex):
                stack.append(neighbor)

        return path
