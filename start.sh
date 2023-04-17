#!/bin/sh
ssh root@10.128.0.20 '/usr/local/hadoop/bin/hdfs --daemon start datanode'
ssh root@10.128.0.21 '/usr/local/hadoop/bin/hdfs --daemon start datanode'

/usr/local/hadoop/sbin/start-dfs.sh
/usr/local/spark/sbin/start-all.sh
