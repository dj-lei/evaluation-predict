#!/bin/sh
LANG=en_US.UTF-8

# 杀掉所有python进程
# kill -9 $(ps -e|grep python |awk '{print $1}')

# 进入工程目录
project_path=/home/ml/evaluation-predict
cd ${project_path}

# 创建日志
if [ ! -d ./log/ ];
then
    mkdir -p log;
    touch ./log/django.log
    touch ./log/cron.log
fi

sleep 1s
# 启动django服务
# nohup /usr/bin/python3 -u manage.py runserver 0.0.0.0:8001 > ./log/django.log 2>&1 &

sleep 3s
# 执行全车型建模
cd api_valuate
nohup /usr/bin/python3 -u start.py > ../log/cron.log 2>&1


