import csv
import re
import pandas as pd
def washfile(file):    #传入文件路径
    with open(file,'r',encoding="ANSI") as f:  #utf-8看需要
        filelist=[]
        reader = csv.reader(f)
        fieldnames = next(reader)#获取数据的第一列，作为后续要转为字典的键名 生成器，next方法获取
        # print(fieldnames)
        csv_reader = csv.DictReader(f,fieldnames=fieldnames) #self._fieldnames = fieldnames   # list of keys for the dict 以list的形式存放键名
        for row in csv_reader:
            d={}
            for k,v in row.items():
                d[k]=v
            testword=d["评论"]
            #去除长度小于6
            if len(testword)<6:
                continue
            else:    
                #去除无中文部分
                RE = re.compile(u'[\u4e00-\u9fa5]', re.UNICODE)
                match = re.search(RE, testword)                        
                if match is None:
                    continue
                else:
                    filelist.append(d)
    #去除重复值
    df=pd.DataFrame(filelist)
    df.drop_duplicates(inplace=True)
    df.to_csv("washedtest1.csv",encoding='ANSI',index=False)
    print("washing finshed")


filename='3.csv'
filelist=washfile(filename)



