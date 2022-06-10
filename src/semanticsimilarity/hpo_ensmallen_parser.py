import os
from ensmallen_graph import EnsmallenGraph


class Hpo2EnsmallenParser:
    """A parser that take in a csv file with edges for the HPO graph and make an Ensmallen
    class that represents the HPO graph
    """

    def __init__(self, edge_path):
        """Constructor

        :param edge_path: a csv file with 'subject' and 'object' columns representing
        the parent and child edges in the HPO graph. Must be comma separated.
        """
        if not os.path.isfile(edge_path):
            raise FileNotFoundError("Could not find HPO edge path at '" + edge_path + "'")
        self._graph = self._read_file(edge_path)
        self._graph_reversed_edges = self._read_file(edge_path,
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
        """Property that returns an ensmallen graph object representing the HPO graph

        :return: ensmallen graph object representing the HPO graph
        """
        return self._graph

    @property
    def graph_reversed_edges(self):
        """Property that returns an ensmallen graph object representing the HPO graph,
        with edges reversed.

        :return: ensmallen graph object representing the HPO graph
        :return:
        """
        return self._graph_reversed_edges
