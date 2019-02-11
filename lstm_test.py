import os
import readdata
import word2vec
import Lstm_Model
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


#文件路径
current_path=os.path.abspath(os.curdir)
test_file_path="D:/urun/Comments_Classifiation-master/data/test.txt"
embedding_model_path="D:/urun/Comments_Classifiation-master/data/embedding_64.bin"
train_data_path="D:/urun/Comments_Classifiation-master/data/lstm/Takeway_lstm/takeway_training_params.pickle"
# train_data_path="D:/urun/Comments_Classifiation-master/data/lstm/training_params.pickle"


#模型超参
class config():
    test_sample_percentage=0.03
    num_labels=2
    embedding_size=64
    dropout_keep_prob=1
    batch_size=64
    num_epochs=80
    max_sentences_length=40
    num_layers=2
    max_grad_norm=5
    l2_rate=0.0001


def get_lstm_result():
    if not os.path.exists(embedding_model_path):
        print("word2vec model is not found")

    if not os.path.exists(train_data_path):
        print("train params is not found")

    params = readdata.loadDict(train_data_path)
    train_length = int(params['max_sentences_length'])



    test_sample_lists = readdata.get_cleaned_list(test_file_path)
    test_sample_lists,max_sentences_length = readdata.padding_sentences(test_sample_lists,padding_token='<PADDING>',padding_sentence_length=train_length)
    test_sample_arrays=np.array(word2vec.get_embedding_vector(test_sample_lists,embedding_model_path))
    testconfig=config()
    testconfig.max_sentences_length=max_sentences_length


    sess=tf.InteractiveSession()
    lstm=Lstm_Model.TextLSTM(config=testconfig)

    saver = tf.train.Saver()
    saver.restore(sess, "D:/urun/Comments_Classifiation-master/data/lstm/text_model")

    #定义测试函数
    def test_step(x_batch):
        feed_dict={
            lstm.input_x:x_batch,
            lstm.dropout_keep_prob:testconfig.dropout_keep_prob
        }
        predictions,scores=sess.run(
            [lstm.predictions,lstm.softmax_result],
            feed_dict=feed_dict
        )
        return (predictions,scores)

    predictions, scores=test_step(test_sample_arrays)
    return np.array(predictions)
    #print("(0->neg & 1->pos)the result is:")
    #print(predictions)
    #print("********************************")
    #print("the scores is:")
    #print(scores)
