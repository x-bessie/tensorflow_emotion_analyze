# -*- coding: utf-8 -*-
"""
Created on Tues Nov  27  15:03:21 2018
底层获取文章测试数据
@author: Lina
"""
import requests
import re
import pymysql
import json
import codecs
from database.textrank import querydb,queryId,querytitle
from textrank4zh import TextRank4Keyword, TextRank4Sentence
from readdata import clean_str,split_str

def connectdb():
    mysql_server='localhost'
    name='root'
    password='123456'
    mysql_db='text'
    db=pymysql.connect(mysql_server,name,password,mysql_db)
    return db 

#请求接口参数
def post_config():
    home='http://hd.article.yunrunyuqing.com:38015/web/search/common/article/select?'
    #token
    token='token=7f2a7d48-23ae-4d2f-b20c-7c5ebfb47d18'
    #过滤参数
    rows='&rows=10'
    fl='&fl=Content,Title'
    #是否原创
    Original='&Original=1'
    #检索id_array
    id_array="&id_array="
    #word
    word="&word=致敬"
    #sitetype
    sitetype="&sitetype=1"
    #时间
    starttime="&starttime=20180201"
    endtime="&endtime=20180301"
    #语言
    language="&language=1"
    #post请求
    url=home+token+rows+fl+word+sitetype+starttime+endtime
    # print(url)
    return url

#请求接口获取待分析的内容到数据库中
def post_url():
    bottom_url=requests.post(post_config()).json()
    total=bottom_url['total']

    for i in range(len(bottom_url['results'])):
        dict=bottom_url['results'][i]
        Title=dict['Title']
        Content=dict['Content']
        # print('{} {}\n'.format(ID,Content))
        db=connectdb()
        cursor=db.cursor()
        cursor.execute("insert into textrank (content,title) values(%s,%s)",(str(Content),str(Title)))
        db.commit()
        db.close()


#摘取重点句子的方法,以及摘取之后的分词
def sentence():
 #读取推文文章内容
    db=connectdb()
    content=querydb(db) #文章内容
    IDs=queryId(db)  #文章的id
    all_text_sentence=[]
    # print(content)
    for i in content:
        str_text = str(i)
        tr4s = TextRank4Sentence()
        tr4s.analyze(text=str_text, lower=True, source = 'all_filters')
        # print( '摘要：' )
        contant=[]
        
        for item in tr4s.get_key_sentences(num=4):
            # print(item.sentence)
            contant.append(item.sentence)
        # print(contant)

        str_contant=str(contant)
        str_content=clean_str(split_str(str_contant))
        all_text_sentence.append(str_content)
    # print(all_text_sentence)

    # for i in range(len(content)):
    #     # 循环取出id
    #     ids=IDs[i]
    #     print('{}\n'.format(contant))
    #     cursor=db.cursor()
    #     cursor.execute("update textrank set sentence=%s where id=%s",(str(contant),str(ids)))
    #     db.commit()
    #     db.close()
    return all_text_sentence

#整篇文章的分词，不摘取句子的判断
def all_sentence():
    db=connectdb()
    content=querydb(db) #文章内容
    Contant=[]

    for i in content:
        str_content=str(i)
        str_content=clean_str(split_str(str_content))
        Contant.append(str_content)
    return Contant

def title_artical():
    db=connectdb()
    content=querytitle(db) #文章内容
    Contant=[]

    for i in content:
        str_content=str(i)
        str_content=clean_str(split_str(str_content))
        Contant.append(str_content)
    print(Contant)

    return Contant

# post_url()
# post_config()
sentence()
# all_sentence()
# title_artical()