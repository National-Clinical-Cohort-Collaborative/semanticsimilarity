# from pyspark.sql import functions as F
import os
# from pyspark.sql import SQLContext
# import pyspark
from ensmallen_graph import EnsmallenGraph
import shutil


class Hpo2EnsmallenParser:
    """
    My great documention
    """
    def __init__(self, edge_path):
        if not os.path.isfile(edge_path):
            raise FileNotFoundError("Could not find HPO edge path at '" + edge_path + "'")
        self._graph = self._read_file(edge_path)

    def _read_file(self, edges_file):
        return EnsmallenGraph.from_unsorted_csv(
            edge_path=edges_file,
            directed=False,
            sources_column="subject",
            destinations_column="object",
            edge_separator="  "
            )

    @property
    def graph(self):
        return self._graph
