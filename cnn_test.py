import tensorflow as tf
import numpy as np
import readdata
import word2vec
import os
import Cnn_Model
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim
from visual import show_emtion 
from readdata import get_all_data_from_file
test_file_path="D:/urun/Comments_Classifiation-master/data/test.txt"
#酒店语料模型测试
# train_data_path="D:/urun/Comments_Classifiation-master/data/cnn/training_params.pickle"
#1.6w外卖语料模型测试
train_data_path="D:/urun/Comments_Classifiation-master/data/cnn/Takeway_cnn/takeway_training_params.pickle"

embedding_model_path="D:/urun/Comments_Classifiation-master/data/embedding_64.bin"

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

def get_cnn_result():
    if not os.path.exists(embedding_model_path):
        print("word2vec model is not found")

    if not os.path.exists(train_data_path):
        print("train params is not found")

    params = readdata.loadDict(train_data_path)
    train_length = int(params['max_sentences_length'])

#写入文件，处理文件
    test_sample_lists = readdata.get_cleaned_list(test_file_path)
    test_sample_lists,max_sentences_length = readdata.padding_sentences(test_sample_lists,padding_token='<PADDING>',padding_sentence_length=train_length)
    test_sample_arrays=np.array(word2vec.get_embedding_vector(test_sample_lists,embedding_model_path))
    testconfig=config()
    testconfig.max_sentences_length=max_sentences_length

    sess=tf.InteractiveSession()
    cnn=Cnn_Model.TextCNN(config=testconfig)

    #加载参数
    saver = tf.train.Saver()
    saver.restore(sess, "D:/urunD:/urun/Comments_Classifiation-master/data/cnn/text_model")

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
        return (predictions,scores)


    #拿到结果
    predictions,scores=test_step(test_sample_arrays)
    return np.array(predictions)
    # print("(0->neg & 1->pos)the result is:")
    # print(predictions) #每一行的句子判断是以断句判断一次，判断出的结果为矩阵结果[0 0 0]
    # print("********************************")
    # print("the scores is:")
    # print(scores)  #对每一个断句进行判断
    # print(scores.shape)

# get_cnn_result()

