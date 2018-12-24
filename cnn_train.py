import os
import readdata
import word2vec
import Cnn_Model
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


#文件路径
current_path=os.path.abspath(os.curdir)

#酒店模型
data_path="D:/urun/Comments_Classifiation-master/data"
positive_file_path="D:/urun/Comments_Classifiation-master/data/pos.txt"
negative_file_path="D:/urun/Comments_Classifiation-master/data/neg.txt"
embedding_model_path="D:/urun/Comments_Classifiation-master/data/embedding_64.bin"
train_data_path="D:/urun/Comments_Classifiation-master/data/cnn/training_params.pickle"
log_path="/home/urun/Documents/project/Comments_Classifiation-master/summary/cnn"

#1.6w外卖语料模型
# data_path="D:/urun/Comments_Classifiation-master/data"
# positive_file_path="D:/urun/Comments_Classifiation-master/data/Takeway/pos.txt"
# negative_file_path="D:/urun/Comments_Classifiation-master/data/Takeway/neg.txt"
# embedding_model_path="D:/urun/Comments_Classifiation-master/data/embedding_64.bin"
# train_data_path="D:/urun/Comments_Classifiation-master/data/cnn/Takeway_cnn/takeway_training_params.pickle"
# log_path="/home/urun/Documents/project/Comments_Classifiation-master/summary/Takeway_cnn"


#模型超参
class config():
    test_sample_percentage=0.03
    num_labels=2
    embedding_size=64
    filter_sizes=[2,3,4]
    num_filters=128
    dropout_keep_prob=0.5
    l2_reg_lambda=0.1
    batch_size=32
    num_epochs=15
    max_sentences_length=0
    lr_rate=1e-3


#加载数据
all_sample_lists,all_label_arrays,max_sentences_length=readdata.get_all_data_from_file(positive_file_path,negative_file_path,force_len=40)
all_sample_arrays=np.array(word2vec.get_embedding_vector(all_sample_lists,embedding_model_path))
del all_sample_lists
print("sample.shape = {}".format(all_sample_arrays.shape))
print("label.shape = {}".format(all_label_arrays.shape))
trainconfig=config()
trainconfig.max_sentences_length=max_sentences_length
testconfig=config()
testconfig.max_sentences_length=max_sentences_length
testconfig.dropout_keep_prob=1.0


#存储训练参数
params={"num_labels":trainconfig.num_labels,"max_sentences_length":max_sentences_length}
readdata.save(params,train_data_path)

#打乱样本顺序
np.random.seed(10)
random_index=np.random.permutation(np.arange(len(all_label_arrays)))
random_sample_arrays=all_sample_arrays[random_index]
del all_sample_arrays
random_label_arrays=all_label_arrays[random_index]
#按比例抽取测试样本
num_tests=int(trainconfig.test_sample_percentage*len(all_label_arrays))
del all_label_arrays,random_index
test_sample_arrays=random_sample_arrays[:num_tests]
train_sample_arrays=random_sample_arrays[num_tests:]
del random_sample_arrays
test_label_arrays=random_label_arrays[:num_tests]
train_label_arrays=random_label_arrays[num_tests:]
del random_label_arrays
print("Train/Test split: {:d}/{:d}".format(len(train_label_arrays), len(test_label_arrays)))

#开始训练
with tf.Graph().as_default():
    sess=tf.Session()
    with sess.as_default():
        cnn=Cnn_Model.TextCNN(config=trainconfig)

        #初始化参数
        train_writer = tf.summary.FileWriter(log_path + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(log_path + '/test')
        step_num=0
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver()


        #定义训练函数
        def train_step(x_batch,y_batch,lr_rate):
            feed_dict={
                cnn.input_x:x_batch,
                cnn.input_y:y_batch,
                cnn.dropout_keep_prob:trainconfig.dropout_keep_prob,
                cnn.learning_rate:lr_rate
            }
            summary,loss,accuracy,_=sess.run(
                [cnn.merged,cnn.loss,cnn.accuracy,cnn.train_op],
                feed_dict=feed_dict
            )
            return (summary,loss,accuracy)

        #定义测试函数
        def test_step(x_batch,y_batch):
            feed_dict={
                cnn.input_x:x_batch,
                cnn.input_y:y_batch,
                cnn.dropout_keep_prob:testconfig.dropout_keep_prob
            }
            summary,loss,accuracy=sess.run(
                [cnn.merged,cnn.loss,cnn.accuracy],
                feed_dict=feed_dict
            )

            return (summary,loss,accuracy)

        #生成批数据
        batches=readdata.batch_iter(
            list(zip(train_sample_arrays, train_label_arrays)),trainconfig.batch_size,trainconfig.num_epochs)


        #正式开始训练啦
        for batch in batches:
            step_num += 1
            x_batch,y_batch=zip(*batch)
            summary,loss, accuracy=train_step(x_batch,y_batch,config.lr_rate)
            if step_num % 20 == 0:
                train_writer.add_summary(summary,step_num)
                #print("For train_samples: step %d, loss %g, accuracy %g" % (step_num,loss,accuracy))
                summary,loss, accuracy = test_step(test_sample_arrays, test_label_arrays)
                #print("Testing loss: %g,Testing accuracy: %g" % (loss, accuracy))
                test_writer.add_summary(summary, step_num)

        #_,loss, accuracy = test_step(test_sample_arrays, test_label_arrays)
        #print("Testing loss: %g,Testing accuracy: %g" % (loss, accuracy))

        saver.save(sess,"D:/urun/Comments_Classifiation-master/data/cnn/text_model")
        train_writer.close()
        test_writer.close()
    