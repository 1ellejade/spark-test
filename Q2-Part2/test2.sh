#!/bin/bash
source ../env.sh
/usr/local/hadoop/bin/hdfs dfs -rm -r /Q2-Part2/input/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /Q2-Part2/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../shot_logs.csv /Q2-Part2/input/
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./subq_2.py hdfs://$SPARK_MASTER:9000/Q2-Part2/input/
