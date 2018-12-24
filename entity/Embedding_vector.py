import gensim
import numpy as np
class Embedding_vector(object):

    @staticmethod
    def load_vector():
        print("loading")
        embedding_model_path="D:/urun/Comments_Classifiation-master/data/embedding_64.bin"
        model=gensim.models.KeyedVectors.load_word2vec_format(embedding_model_path,binary=True)
        print("loading finished...")
        return model

