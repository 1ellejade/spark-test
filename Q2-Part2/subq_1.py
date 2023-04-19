from __future__ import print_function
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import SparkSession
import sys
from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler

if __name__ == "__main__":
  spark = SparkSession\
    .builder\
    .appName("NBAKMeans")\
    .getOrCreate()

  df=spark.read\
    .format("csv")\
    .option("inferSchema","true")\
    .option("header","true")\
    .load(sys.argv[1]) 

  player_list=list(df.toPandas()['player_name'].unique())

  for player in player_list:
    print(player, end=': ')
    new_df=df.filter(f"player_name == '{player}'")
    input_cols=['SHOT_CLOCK', 'SHOT_DIST', 'CLOSE_DEF_DIST',]
    vec_assembler=VectorAssembler(inputCols=input_cols, outputCol="zones", handleInvalid = "skip")
    final_df=vec_assembler.transform(new_df)

    kmeans = KMeans(featuresCol='zones').setK(4).setSeed(1)
    model = kmeans.fit(final_df)
    predictions = model.transform(final_df)

    centers = model.clusterCenters()
    for center in centers:
        print(center, end=' ')
    print('\n')

  spark.stop()
