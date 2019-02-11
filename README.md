# tensorflow_emotion_analyze
判断微博情绪正负面


配置运行环境

Linux上：

Ubantu 14.04 LTS

Anaconda 5.1.0 

tensorflow 1.11.0

python 3.6

jieba

gensim

mysql-connector-python

Windows：

win10

Anaconda 5.1.0

python3.6

tensorflow 1.11.0  gpu版本

gensim

mysql-connector-python

jieba

程序在linux和windows都可运行。


项目所用算法

CNN 卷积神经网络

LSTM 长短期记忆


Word2vec

项目使用的词向量：embedding_64.bin(1.5G)

训练语料：百度百科800w条 20G+搜狐新闻400w条 12G+小说：90G左右

模型参数：window=5 min_count=5 size=64


下载地址：
链接：https://pan.baidu.com/s/1L5c_LrIi89dOA9HFxP2zjw 
提取码：20pj 

项目功能

1.根据数据库提供的唯一推文ID请求底层，拿到底层中的conten进行情绪正负面判断
2.判断的结果 0 -未判断  1-正 2-负 3-中
3.将判断好的结果更新到对应的数据库
具体可看 tensor判断情绪正负面（微博）.xmind


程序介绍

./ten_threading.py   入口

./man_tensorflow.py  

./man_test.py       测试文件

./data              包含训练好的模型，词向量

./database          数据库配置，数据库操作

./entity            运行的类

./summy             运行的log

./readdata.py       对目标词的处理，分词，padding等

./cnn_test_sql.py   CNN算法判断的函数，包括判断的结果处理

./Cnn_Model.py      Cnn模型

./cnn_train.py      Cnn模型的训练

./Lstm_Model.py     LSTM模型

./lstm_test_sql     lstm算法判断函数，包括判断的结果处理

./lstm_train.py     LSTM模型的训练

./getapi            对底层的请求


程序的不足

1.短篇幅的语句，短语句评论达准确率90%。对长篇文章的判断准确率下降，可能需要更好的长篇语料
2.读取语料的方式最好为读文件的方式，现在读单个文本的方式