import os
from ensmallen_graph import EnsmallenGraph


class Hpo2EnsmallenParser:
    """
    My great documention
    """

    def __init__(self, edge_path):
        if not os.path.isfile(edge_path):
            raise FileNotFoundError("Could not find HPO edge path at '" + edge_path + "'")
        self._graph = self._read_file(edge_path)
        self._graph_reversed_edges = self._read_file_reverse(edge_path,
                                                             sources_columns="object", destinations_column="subject")

    def _read_file(self, edges_file, sources_columns="subject", destinations_column="object"):
        return EnsmallenGraph.from_unsorted_csv(
            edge_path=edges_file,
            directed=True,
            sources_column=sources_columns,
            destinations_column=destinations_column,
            edge_separator=","
            )

    @property
    def graph(self):
        return self._graph

    @property
    def graph_reversed_edges(self):
        return self._graph_reversed_edges
