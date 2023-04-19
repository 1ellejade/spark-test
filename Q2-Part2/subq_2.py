from __future__ import print_function

import sys
import math
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, MapType

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
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

  players_wanted=['james harden', 'chris paul', 'stephen curry', 'lebron james']

  for player in players_wanted:                            
    df.filter(f"player_name == '{player}'")

    print(player, end=': ')
    new_df=df.filter(f"player_name == '{player}'")
    input_cols=['SHOT_CLOCK', 'SHOT_DIST', 'CLOSE_DEF_DIST',]
    vec_assembler=VectorAssembler(inputCols=input_cols, outputCol="zones", handleInvalid = "skip")
    final_df=vec_assembler.transform(new_df)
    final_df

    kmeans = KMeans(featuresCol='zones').setK(4).setSeed(1)
    model = kmeans.fit(final_df)
    predictions = model.transform(final_df)

    centers = model.clusterCenters()

    df2=final_df.rdd.map(lambda x: ((0 if x['SHOT_RESULT']=='missed' else 1), x['zones'])).toDF(['SHOT_RESULT', 'zones'])
    

    centers_list=[]       
    shot_list=[0,0,0,0] #to put numbers to the assigned zones
    hit_list=[0,0,0,0]

    for center in centers:
      centers_list.append(list(center))

    def euclidean(x, y): 
      dist=0
      for i in range(1,len(x)): 
        dist+=((float(x[i])-float(y[i]))**2)

      dist=math.sqrt(dist)
      return dist

    zone_df = df2.select(col('zones'), col('SHOT_RESULT')).toPandas()
    for i in range(len(zone_df)):
      z=zone_df['zones'][i]
      r=zone_df['SHOT_RESULT'][i]

      distances=[0,0,0,0]
      for c in range(len(centers_list)):
        distances[c]=euclidean(list(z), centers_list[c])

      min_value=min(distances)
      assignment=distances.index(min_value)
      shot_list[assignment]=shot_list[assignment]+1
      hit_list[assignment]=hit_list[assignment]+int(r)

    #after hits are aggregated, calc rate
    rate_list=[0,0,0,0]
    for i in range(4):
      rate_list[i]=hit_list[i]/shot_list[i]
      
    max_rate=max(rate_list)
    best_z=rate_list.index(max_rate)
    print(centers_list[best_z])
  
  spark.stop()
