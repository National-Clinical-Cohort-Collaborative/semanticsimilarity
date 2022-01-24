# from pyspark.sql import functions as F
import os
from pyspark.sql import SQLContext
import pyspark
from ensmallen_graph import EnsmallenGraph
import shutil


class Hpo2EnsmallenParser:
    """
    My great documention
    """
    def __init__(self, node_path, edge_path):
        if not os.path.isfile(node_path):
            raise FileNotFoundError("Could not find HPO node path at '" + node_path + "'")
        if not os.path.isfile(edge_path):
            raise FileNotFoundError("Could not find HPO edge path at '" + edge_path + "'")
        self._graph = self._read_file(edge_path)

    def _read_file(edges_file):
        inp = edges_file
        fs = inp.filesystem()
        filename = edges_file
        with fs.open(filename, 'rb') as f:
            with open(filename, "wb") as g:
                shutil.copyfileobj(f, g)

                print(os.listdir())
        with open(filename, 'r') as content_file:
            print(content_file.read())

        return EnsmallenGraph.from_unsorted_csv(
            edge_path=filename,
            directed=False,
            sources_column="subject",
            destinations_column="object")

    @property
    def graph(self):
        return self._graph

