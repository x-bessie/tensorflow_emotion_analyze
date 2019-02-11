from  mixed_cnn_lstm_test_sql import get_mixed_result
import numpy as np
import pymysql
from database.selectapi import connectdb,querydb,updatescore,queryId
import datetime
from  multiprocessing  import Process,Queue,Pool
from multiprocessing.dummy import Pool as ThreadPool
import time


from entity.Embedding_vector import Embedding_vector
model=Embedding_vector.load_vector()

def my_func(a):
    return (a[0]-a[-1])

def score_abs():
    c=0
    return c

def Judgement(model):
    start = datetime.datetime.now()
    prediction = np.array([])
    db=connectdb()
    print("********************欢迎使用舆情分析工具***********************")
    prediction= get_mixed_result(model)

#判断
    predictions=prediction[0].tolist()
    change_predictions=predictions[0]   #判断值
    print(change_predictions)

#判断值
    score_pred=(prediction[1])[0]
    print(score_pred)  #判断的值

    deal_score=np.apply_along_axis(my_func,1,score_pred)
    deal_scoree=np.fabs(deal_score)

    print(deal_scoree) #已经处理过的单一值

    float_scoree=deal_scoree.tolist()

#取到唯一的插入对应的ID,已经处理过
    ID=queryId(db)  
    content=[]

#插入时间
    updatejudgement=datetime.datetime.now()
    # for i in range(len(change_predictions)):
    #     if change_predictions[i]==0:
    #         change_predictions[i]=2
    #     else:
    #         change_predictions[i]=1
    #     judges=change_predictions[i]
    #     content.append(judges)

    #     list_content=content[i]
    #     score=float_scoree[i]
        
    #     ids=ID[i]
    #     print('{} {}\n'.format(list_content,score))
    #     cursor=db.cursor()
    #     cursor.execute("update case_dispose set Judge=%s,JudgeScore=%s where ID=%s",(str(list_content),str(score),str(ids)))
    #     db.commit()


Judgement(model)