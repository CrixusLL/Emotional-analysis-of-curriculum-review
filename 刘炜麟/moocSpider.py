import re
import json
import requests
import threading
import pandas as pd
from queue import Queue
from bs4 import BeautifulSoup

class icourse163_spider():
    def __init__(self):
        self.csrfKey="3f6ad3fddede4401ad857975bf2825b4"
        self.cookie="__yadk_uid=j9fzEkUxWc39zSKNll1L4W1KwaS7FrVw; MOOC_PRIVACY_INFO_APPROVED=true; hb_MA-A976-948FFA05E931_source=mail.qq.com; WM_TID=%2BlV3wZ9f55hEVAUAVBc7O6M6WDSjSDXT; videoResolutionType=2; videoRate=1.5; hasVolume=false; videoVolume=0; Hm_lvt_77dc9a9d49448cf5e629e5bebaa5500b=1615714695,1615877501,1616474414,1616757763; WM_NI=Hzb7JZEa2RSE%2BDqvquSETQ0fW7njTqdPyFGEJMOfBzSoEpDjuGo3zjlhqesimpA40LgZCqtF9BLg4RJEjo9f%2B8lijhMFjDw2CqhWYdzkwbVbo5oS%2FLTYalxPWau%2BSvhPclE%3D; WM_NIKE=9ca17ae2e6ffcda170e2e6eea4aa47818cf986ce63b48a8fb6c44f978b8bafb66898b7bcaad26b9bb4fdbbf52af0fea7c3b92a8ca6aed8c95287bffeb3d24fe99cbf97f47f919b9cb2b36e95e9a28bd77ba890b7a7c64f8a938187c17291b18b91c762b799a1d4f962b0b8bf86f263afb08fd5b348a5868d92f74efb9e8488b75a8a9ba690bc6ee9e8a48ae9648f9aa6a5f63a81afa8b8dc70bab5b983d56aadeba1b4ae489887af8dfb5c95bfbbccd27eac90ad8dc837e2a3; CLIENT_IP=220.115.183.130; Hm_lpvt_77dc9a9d49448cf5e629e5bebaa5500b=1616775603; "
        # self.headers={
        #     "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36",
        #     "edu-script-token": "4680367f4df140fb885125ae3ed7f1bc"
        # }
        self.headers={
            "cookie":"EDUWEBDEVICE=5cb2b29edbb94deb8d0277c44e917bcb; __yadk_uid=j9fzEkUxWc39zSKNll1L4W1KwaS7FrVw; MOOC_PRIVACY_INFO_APPROVED=true; hb_MA-A976-948FFA05E931_source=mail.qq.com; WM_TID=%2BlV3wZ9f55hEVAUAVBc7O6M6WDSjSDXT; videoResolutionType=2; videoRate=1.5; hasVolume=false; videoVolume=0; WM_NI=Hzb7JZEa2RSE%2BDqvquSETQ0fW7njTqdPyFGEJMOfBzSoEpDjuGo3zjlhqesimpA40LgZCqtF9BLg4RJEjo9f%2B8lijhMFjDw2CqhWYdzkwbVbo5oS%2FLTYalxPWau%2BSvhPclE%3D; WM_NIKE=9ca17ae2e6ffcda170e2e6eea4aa47818cf986ce63b48a8fb6c44f978b8bafb66898b7bcaad26b9bb4fdbbf52af0fea7c3b92a8ca6aed8c95287bffeb3d24fe99cbf97f47f919b9cb2b36e95e9a28bd77ba890b7a7c64f8a938187c17291b18b91c762b799a1d4f962b0b8bf86f263afb08fd5b348a5868d92f74efb9e8488b75a8a9ba690bc6ee9e8a48ae9648f9aa6a5f63a81afa8b8dc70bab5b983d56aadeba1b4ae489887af8dfb5c95bfbbccd27eac90ad8dc837e2a3; NTESSTUDYSI=3f6ad3fddede4401ad857975bf2825b4; Hm_lvt_77dc9a9d49448cf5e629e5bebaa5500b=1615877501,1616474414,1616757763,1616812308; Hm_lpvt_77dc9a9d49448cf5e629e5bebaa5500b=1616819938",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36",
            "edu-script-token": "3f6ad3fddede4401ad857975bf2825b4"
        }
        self.school_urls=Queue()
        self.school_ids=Queue()
        self.course_comments=Queue()
        self.course_ids=[]
        self.evaluation=[]
        self.school_len=0
        self.count=0

    def __update_cookie(self,res):  #更新cookie
        cookies=res.cookies.get_dict()
        try:
            edu_device=cookies["EDUWEBDEVICE"]
            self.csrfKey=cookies["NTESSTUDYSI"]
            self.headers["edu-script-token"]=self.csrfKey
            self.headers["cookie"]=self.cookie+"EDUWEBDEVICE="+edu_device+"; "+"NTESSTUDYSI="+self.csrfKey
        except KeyError:
            print("无法获取CSRF token！")

    def __request_get(self,url):    #get请求
        headers={
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36"
        }
        res=requests.get(url,headers=headers)
        res.encoding='utf-8'
        soup=BeautifulSoup(res.text,'html.parser')
        #self.__update_cookie(res)
        return soup

    def __get_school_urls(self):    #获取所有学校URL
        soup=self.__request_get("https://www.icourse163.org/university/view/all.htm#/")
        tags=soup.find_all("a",class_="u-usity f-fl")
        self.school_len=len(tags)
        for tag in tags:
            self.school_urls.put("https://www.icourse163.org"+tag["href"])
            print("成功提取 "+tag.find("img")["alt"]+" URL")
        print("\n提取高校URL完成！")
    
    def __get_school_ids(self):     #获取所有学校id
        while True:
            url=self.school_urls.get()
            soup=self.__request_get(url)
            id_=re.findall('schoolId = "(.+)"',soup.get_text())[0]
            self.school_ids.put(id_)
            #print("成功提取ID "+id_)
            self.school_urls.task_done()
    
    def __get_course_ids(self):     #获取所有课程id
        url="https://www.icourse163.org/web/j/courseBean.getCourseListBySchoolId.rpc"
        count=0
        while True:
            self.count+=1
            id_=self.school_ids.get()
            data={
                "schoolId":id_,
                "p":1,
                "psize":20,
                "type":1,
                "courseStatus":30
                }
            page=0
            while True:    #请求每个大学所有页的课程id
                page+=1
                data["p"]=page
                flag=False
                while flag==False:
                    try:
                        r=requests.post(url+"?csrfKey="+self.csrfKey,headers=self.headers,data=data)
                        flag=True
                    except:     #防止服务器端无响应
                        print("Connection Error! Retrying......")
                        continue
                res=json.loads(r.content.decode())
                if res["result"]["list"]==None:break
                for i in res["result"]["list"]:
                    self.course_ids.append(i["id"])
            print("提取学校课程id进度：{0}/{1}".format(self.count,self.school_len))
            self.school_ids.task_done()
    
    def __get_course_comments(self):    #获取课程评论
        url="https://www.icourse163.org/web/j/mocCourseV2RpcBean.getCourseEvaluatePaginationByCourseIdOrTermId.rpc"
        while True:
            id_=self.course_ids.get()
            data={
                "courseId":id_,
                "pageIndex":1,
                "pageSize":20,
                "orderBy":3
            }
            page=0
            while True:    #请求每个课程所有页的评论数据
                page+=1
                data["pageIndex"]=page
                flag=False
                while flag==False:
                    try:
                        r=requests.post(url+"?csrfKey="+self.csrfKey,headers=self.headers,data=data)
                        flag=True
                    except:     #防止服务器端无响应
                        print("Connection Error! Retrying......")
                        continue
                res=json.loads(r.content.decode())
                if res["result"]["list"]==[]:break
                for i in res["result"]["list"]:
                    if i["mark"]==3:continue
                    self.course_comments.put(i)
            print("提取课程评论进度：剩余{0}".format(self.course_ids.qsize()))
            self.course_ids.task_done()
    
    def __parse_comments(self):     #生成评论数据列表
        while True:
            data=self.course_comments.get()
            d={"comment":data["content"],"mark":data["mark"]}
            self.evaluation.append(d)
            self.course_comments.task_done()
    
    def save_course_ids(self):      #保存课程id
        for i in range(10):
            with open("./course_ids"+str(i)+".txt","w") as f:
                for d in range(i*1000,(i+1)*1000):
                    try:
                        f.write(str(self.course_ids[d]))
                        f.write("\n")
                    except:
                        break

    def save_csv(self):     #保存所有评论数据
        df=pd.DataFrame(self.evaluation)
        df.to_csv("./reviews/comments9.csv",encoding='utf-8',index=False)
    
    def run_get_courses_ids(self):  #提取所有课程id
        thread_list=[]
        for i in range(1):
            thr=threading.Thread(target=self.__get_school_urls)
            thread_list.append(thr) 
        for i in range(4):
            thr=threading.Thread(target=self.__get_school_ids)
            thread_list.append(thr) 
        for i in range(4):
            thr=threading.Thread(target=self.__get_course_ids)
            thread_list.append(thr) 
        for i in thread_list:
            i.setDaemon(True)
            i.start()

        self.school_urls.join()
        self.school_ids.join()
        print("主线程结束")
        self.save_course_ids()
    
    def run_parse_comments(self):   #提取分析评论数据
        self.course_ids=Queue()
        with open("./course_ids9.txt","r") as f:
            for i in f.read().split("\n"):
                self.course_ids.put(i)
        thread_list=[]
        for i in range(4):
            thr=threading.Thread(target=self.__get_course_comments)
            thread_list.append(thr) 
        for i in range(4):
            thr=threading.Thread(target=self.__parse_comments)
            thread_list.append(thr)   
        for i in thread_list:
            i.setDaemon(True)
            i.start()
        self.course_ids.join()
        self.course_comments.join()
        print("主线程结束")
        self.save_csv()


if __name__=="__main__":
    spider=icourse163_spider()
    spider.run_get_courses_ids()
    #spider.run_parse_comments()
