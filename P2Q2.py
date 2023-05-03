from __future__ import print_function

import sys
import numpy as np

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import isnan, when, count, lit
from pyspark.sql.types import ArrayType, StructField, StructType, StringType, IntegerType, DoubleType

from pyspark.ml.linalg import Vector, Matrices
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.ml.stat import Correlation
from pyspark.mllib.stat import Statistics
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import MulticlassMetrics

if __name__ == "__main__":
  spark = SparkSession\
    .builder\
    .appName("LogReg")\
    .getOrCreate()

  df=spark.read\
    .format("csv")\
    .option("inferSchema","true")\
    .option("header","true")\
    .load('/content/framingham.csv') #*****change this line to sys.argv[1]

#####DATA PREP#####

  df=df.drop('education')
  df=df.na.drop() #checked for null values to make sure this worked
  print('Null values after drop na:')
  df.select([count(when(isnan(c), c)).alias(c) for c in df.columns]).show()

  #casting string dtype to double
  df = df.withColumn('cigsPerDay', df['cigsPerDay'].cast('double'))
  df = df.withColumn('BPMeds', df['BPMeds'].cast('double'))
  df = df.withColumn('totChol', df['totChol'].cast('double'))
  df = df.withColumn('BMI', df['BMI'].cast('double'))
  df = df.withColumn('heartRate', df['heartRate'].cast('double'))
  df = df.withColumn('glucose', df['glucose'].cast('double'))
  
  df = df.withColumn('bias', lit(1.0).cast('double'))

  
#####DATA EXPLORATION#####

  #show value counts for each outcome of TenYearCHD
  print('Target variable outcome distribution:')
  df.groupby('TenYearCHD').count().show()

  #describe dataset
  print('Dataset description:')
  df.describe(['male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose', 'TenYearCHD']).show()


#####FEATURE SELECTION#####
  

  # convert to vector column first
  vector_col = "corr_features"
  assembler = VectorAssembler(inputCols=df.columns[:-1], outputCol=vector_col, handleInvalid ='skip')
  df_vector = assembler.transform(df).select(vector_col)

  # get correlation matrix
  matrix = Correlation.corr(df_vector, vector_col)
  arr = matrix.collect()[0]["pearson({})".format(vector_col)].values

  cols = len(df.columns)-1
  arr=arr.reshape((cols, cols)).tolist()

  rdd = spark.sparkContext.parallelize(arr)
  schema = StructType([
      StructField(df.columns[0], DoubleType(), True),
      StructField(df.columns[1], DoubleType(), True),
      StructField(df.columns[2], DoubleType(), True),
      StructField(df.columns[3], DoubleType(), True),
      StructField(df.columns[4], DoubleType(), True),
      StructField(df.columns[5], DoubleType(), True),
      StructField(df.columns[6], DoubleType(), True),
      StructField(df.columns[7], DoubleType(), True),
      StructField(df.columns[8], DoubleType(), True),
      StructField(df.columns[9], DoubleType(), True),
      StructField(df.columns[10], DoubleType(), True),
      StructField(df.columns[11], DoubleType(), True),
      StructField(df.columns[12], DoubleType(), True),
      StructField(df.columns[13], DoubleType(), True),
      StructField(df.columns[14], DoubleType(), True),  
  ])

  corr_df = spark.createDataFrame(rdd,schema)
  print(corr_df.schema)
  #corr_df.show()

  final_corrs=corr_df.rdd.map(lambda x: x[-1]).collect()
  inds=sorted(range(len(final_corrs)), key=lambda i: final_corrs[i])[-7:]
  print(inds)
  final_cols=[]
  for ind in inds:
    if ind!=14:
      final_cols.append(df.columns[ind])
  print('Most highly correlated features:', final_cols)

#####LOGISTIC REGRESSION#####

  columns=final_cols
  assembler = VectorAssembler(inputCols = columns, outputCol='features', handleInvalid='skip')
  df2 = assembler.transform(df)

  final_df = df2.select('features', 'TenYearCHD')
  train, test = final_df.randomSplit([0.8, 0.2])

  #logistic regression train & test
  lr = LogisticRegression(labelCol="TenYearCHD")
  lrn = lr.fit(train)
  lrn_summary = lrn.summary
  #lrn_summary.predictions.show()
  #lrn_summary.predictions.describe().show()

  pred = lrn.transform(test)
  pred.select('TenYearCHD', 'features',  'rawPrediction', 'prediction', 'probability')
  accuracy = pred.filter(pred.TenYearCHD == pred.prediction).count() / float(pred.count())

  #####EVALUATION#####

  print('Accuracy: ',round(accuracy, 3))
  print('Misclassification: ', round(1-accuracy, 3))

  cm1 = pred.select('TenYearCHD', 'prediction')
  cm1=cm1.withColumn("TenYearCHD",cm1.TenYearCHD.cast(DoubleType()))

  l=[]
  for i in cm1.collect():
    l.append(tuple(i))

  eval_dset = spark.createDataFrame(l, ["raw", "label"])

  evaluator = BinaryClassificationEvaluator()
  evaluator.setRawPredictionCol("raw")
  print('AUC:',evaluator.evaluate(eval_dset))

  predictionAndLabels = spark.sparkContext.parallelize(l)
  metrics = MulticlassMetrics(predictionAndLabels)
  #metrics.confusionMatrix().toArray()
  print('Confusion Matrix:')
  print(metrics.confusionMatrix().toArray())
  print('False positive rate:', end=' ')
  print(round(metrics.falsePositiveRate(1.0),3))
  print('Precision:', end=' ')
  print(round(metrics.precision(1.0),3))
  print('Recall:', end=' ')
  print(round(metrics.recall(1.0),3))
  print('F-score:', end=' ')
  print(round(metrics.fMeasure(0.0, 1.0),3))


  #####LOWER THRESHOLD#####
  thresh_list=[.4, .3, .2, .1]
  for thresh in thresh_list:
    print(f'\nLowering threshold to {thresh}...\n')
    lr_2 = LogisticRegression(labelCol="TenYearCHD")
    lrn_2 = lr_2.fit(train)
    lrn_2.setThreshold(thresh)
    lrn_2_summary = lrn_2.summary

    pred_2 = lrn_2.transform(test)
    pred_2.select('TenYearCHD', 'features',  'rawPrediction', 'prediction', 'probability')
    accuracy = pred_2.filter(pred_2.TenYearCHD == pred_2.prediction).count() / float(pred_2.count())

    print('Accuracy: ',round(accuracy, 3))
    print('Misclassification: ', 1-round(accuracy, 3))

    cm1 = pred_2.select('TenYearCHD', 'prediction')
    cm1=cm1.withColumn("TenYearCHD",cm1.TenYearCHD.cast(DoubleType()))

    l=[]
    for i in cm1.collect():
      l.append(tuple(i))
    
    eval_dset = spark.createDataFrame(l, ["raw", "label"])

    evaluator = BinaryClassificationEvaluator()
    evaluator.setRawPredictionCol("raw")
    print('AUC:',evaluator.evaluate(eval_dset))

    predictionAndLabels = spark.sparkContext.parallelize(l)
    metrics = MulticlassMetrics(predictionAndLabels)
    #metrics.confusionMatrix().toArray()
    print('Confusion Matrix:')
    print(metrics.confusionMatrix().toArray())
    print('Recall:', end=' ')
    print(round(metrics.recall(1.0),3))
