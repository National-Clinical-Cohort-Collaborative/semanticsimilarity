# from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output


@transform_df(
    Output("/UNITE/[RP-2AE058] [N3C Operational] Machine-learning resources for N3C/Semantic_similarity_clustering/src/clustering/annotation_prototype"),
    source_df=Input("SOURCE_DATASET_PATH"),
)
def compute(source_df):
    return source_df
