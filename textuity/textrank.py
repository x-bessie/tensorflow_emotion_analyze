# #-*- encoding:utf-8 -*-
# #摘取重点句子 
# #
# from database.selectapi import connectdb,querydb
# import codecs
# from textrank4zh import TextRank4Keyword, TextRank4Sentence
# import pymysql
# import re
# from lxml import etree
# import requests
# import json
# from readdata import clean_str,split_str
# import datetime

# #获取id数据
# def get_id():
#     db=connectdb()
#     judge_ids=querydb(db)
#     str_judge_ids=str(judge_ids).replace("'","")
#     str_judge_ids=(str(str_judge_ids).replace("[","")).replace("]","")
#     # print(str_judge_ids)
#     return str_judge_ids
# #请求接口参数
# def post_config():
#     home='http://hd.weibo.yunrunyuqing.com:38015/search/common/weibo/select?'
#     #token
#     token='token=7f2a7d48-23ae-4d2f-b20c-7c5ebfb47d18'
#     #过滤参数,底层请求限定为rows=500,字符串限定rows=200左右
#     rows='&rows=200'
#     fl='&fl=Content,id'
#     #是否原创
#     Original='&Original=0'
#     #时间为一周内外的就要设定时间,不然会total=0
#     starttime="&starttime=20181015"
#     endtime="&endtime=20201230"
#     #检索id_array
#     id_array="&id_array="+get_id()
#     #post请求
#     url=home+token+rows+fl+starttime+endtime+Original+id_array
#     # print(url)
#     return url

# #请求接口获取待分析的内容，并分词处理
# def post_url():
#     starttime=datetime.datetime.now()
#     bottom_url=requests.post(post_config()).json()
#     total=bottom_url['total']
    
#     print(total)
#     print(len(bottom_url['results']))

#     Contant=[]
#     if total == 0:
#        print("检查请求时间值endtime") 
#        pass
#     else:    
#         for i in range(len(bottom_url['results'])):
            
#             dict=bottom_url['results'][i]
#             Content=dict['Content']
#             print('{}\n'.format(Content))
#             #分词
#             # str_content=clean_str(split_str(Content))
#             Contant.append(str_content)
#         # print(Contant)
#         # endtime=datetime.datetime.now()
#         # print("post url time and jieba ....")
#         # print(endtime-starttime)
#         return Contant   

# #     def importantsentence():
# # #读取推文文章内容
# #     text = post_url()

# #     tr4w = TextRank4Keyword()

# #     tr4w.analyze(text=text, lower=True, window=2)  # py2中text必须是utf8编码的str或者unicode对象，py3中必须是utf8编码的bytes或者str对象

# #     print( '关键词：' )
# #     for item in tr4w.get_keywords(20, word_min_len=1):
# #         print(item.word, item.weight)
# #         # print(item.word)

# #     print()
# #     print( '关键短语：' )def
# #     for phrase in tr4w.get_keypdefhrases(keywords_num=20, min_occur_num= 2):
# #         print(phrase)def
# # def
# #     tr4s = TextRank4Sentence()def
# #     tr4s.analyze(text=text, lowdefer=True, source = 'all_filters')
# # def
# #     print()def
# #     print( '摘要：' )def
# #     for item in tr4s.get_key_sedefntences(num=4):
# #         # print(item.index, item.weight, item.sentence)  # index是语句在文本中位置，weight是权重
# #         print(item.sentence)


# # post_url()
# post_config()