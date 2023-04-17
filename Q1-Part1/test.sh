#!/bin/bash
source ../env.sh
/usr/local/hadoop/bin/hdfs dfs -rm -r /Q1-Part1/input/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /Q1-Part1/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../parking_sampled.csv /Q1-Part1/input/
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./subq_1.py hdfs://$SPARK_MASTER:9000/Q1-Part1/input/
