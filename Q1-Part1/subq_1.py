from __future__ import print_function
import sys
import math
from operator import add
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, dayofmonth, to_date, regexp_replace
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector


if __name__ == "__main__":
  spark = SparkSession\
    .builder\
    .appName("KMeansExample")\
    .getOrCreate()

  
  dataset = spark.read.format("csv").option("inferSchema", "true").option("header", "true").load('/content/parking_sampled.csv') 
  time_df = dataset.select(col('Violation Time'), col('Summons Number'))
  df2=time_df.rdd.map(lambda x: ((str(int(x['Violation Time'][:2])+12) if x['Violation Time'][4]=='P' else x['Violation Time'][:2], x['Summons Number']))).toDF(['Violation Time','Summons Number'])
  df3 = df2.groupBy('Violation Time').count()
  df3=df3.sort(col('count').desc())

  df3_collect = df3.collect() 

  for row in df3_collect:
    print(row['Violation Time'], row['count'])
  
  spark.stop()
