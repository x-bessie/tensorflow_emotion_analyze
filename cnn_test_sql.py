import tensorflow as tf
import numpy as np
import readdata
import word2vec
import os
import Cnn_Model
import pymysql
from loadsql import query
import getapi

# from entity.Embedding_vector import Embedding_vector
# model=Embedding_vector.load_vector()
test_file_path="D:/urun/Comments_Classifiation-master/data/test.txt"
# train_data_path="D:/urun/Comments_Classifiation-master/data/cnn/training_params.pickle"
train_data_path="D:/urun/Comments_Classifiation-master/data/cnn/Takeway_cnn/takeway_training_params.pickle"
# embedding_model_path="D:/urun/Comments_Classifiation-master/data/embedding_64.bin"

class config():
    test_sample_percentage=0.03
    num_labels=2
    embedding_size=64
    filter_sizes=[2,3,4]
    num_filters=128
    dropout_keep_prob=1
    l2_reg_lambda=0.1
    batch_size=32
    num_epochs=15
    max_sentences_length=0
    lr_rate=1e-3

def get_cnn_result(model):
    # if not os.path.exists(embedding_model_path):
    #     print("word2vec model is not found")
    
    if not os.path.exists(train_data_path):
        print("train params is not found")

    params = readdata.loadDict(train_data_path)
    train_length = int(params['max_sentences_length'])

#写入文件，处理文件
    # mysql_server='localhost'
    # name='root'
    # password='123456'
    # mysql_db='tensor'
    # db=pymysql.connect(mysql_server,name,password,mysql_db)
    
#分词处理
    # test_sample_lists = readdata.get_cleaned_list(test_file_path) ###1.训练模型用
    test_sample_lists=readdata.get_query_list_cnn(db)  # #测试用
    #插入底层的分词处理
    # test_sample_lists=getapi.post_url()
    test_sample_lists,max_sentences_length = readdata.padding_sentences(test_sample_lists,padding_token='<PADDING>',padding_sentence_length=train_length)
    #改前方法
    # test_sample_arrays=np.array(word2vec.get_embedding_vector(test_sample_lists,embedding_model_path))
    #改后
    test_sample_arrays=np.array(word2vec.get_embedding_vector(test_sample_lists,model))

    testconfig=config()
    testconfig.max_sentences_length=max_sentences_length

    sess=tf.InteractiveSession()
    cnn=Cnn_Model.TextCNN(config=testconfig)
    
    
      
    #加载参数
    # tf.get_variable_scope().reuse_variables()
    saver = tf.train.Saver()
    saver.restore(sess, "E:/资料/Comments_Classifiation-master/data/cnn/text_model")
     
    #定义测试函数,可以给出相对应的预测还有分数。sess.run:变量的赋值和计算
     
    
    def test_step(x_batch):
        
        feed_dict={
            cnn.input_x:x_batch,
            cnn.dropout_keep_prob:1.0
            }
        predictions,scores=sess.run(
            [cnn.predictions,cnn.softmax_result],
            feed_dict=feed_dict
            )
        sess.close()
        return (predictions,scores)
   
    #拿到结果
    predictions,scores=test_step(test_sample_arrays)

    return predictions,scores
    # print("(0->neg & 1->pos)the result is:")
    # print(predictions) 
    # print("********************************")
    # print("the scores is:")
    # print(scores)  
# get_cnn_result(model)