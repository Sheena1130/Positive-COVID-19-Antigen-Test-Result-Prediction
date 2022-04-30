#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import requests
import json
import math
import os
from datetime import date, datetime, timedelta

from itertools import groupby
import bs4
import re


# In[ ]:


def get_tested_positive(save_file = True):
    
    # fetch data from official website
    response = requests.get("https://covid-19.nchc.org.tw/api/covid19?CK=covid-19@nchc.org.tw&querydata=4001&limited=TWN").text
    
    # transfer json to pandas
    jsonData = json.loads(response)
    df = pd.DataFrame.from_dict(jsonData)
    
    # rename columns
    df.rename(columns = {"id":"ID","a01":"iso_code","a02":"洲名","a03":"國家","a04":"日期","a05":"總確診數","a06":"新增確診數","a07":"七天移動平均新增確診數","a08":"總死亡數","a09":"新增死亡數","a10":"七天移動平均新增死亡數","a11":"每百萬人確診數","a12":"每百萬人死亡數","a13":"傳染率","a14":"新增檢驗件數","a15":"總檢驗件數","a16":"每千人檢驗件數","a17":"七天移動平均新增檢驗件數","a18":"陽性率","a19":"每確診案例相對檢驗數量","a20":"疫苗總接種總劑數","a21":"疫苗總接種人數","a22":"疫苗新增接種劑數","a23":"七天移動平均疫苗新增接種劑數","a24":"每百人接種疫苗劑數","a25":"每百人接種疫苗人數","a26":"疫情控管指數","a27":"總人口數","a28":"中位數年紀","a29":"70歲以上人口比例","a30":"平均壽命","a31":"解除隔離數","a32":"解封指數"}, inplace = True)
    
    # keep useable columns and rename into English
    df = df[["日期", "新增確診數"]]
    df = df.rename(columns = {"日期":"survey_date", "新增確診數":"tested_positive_num"})
    
    # sort by survey date
    df["survey_date"] = pd.to_datetime(df["survey_date"], format = "%Y-%m-%d")
    df = df.sort_values(by = "survey_date")
    df = df.reset_index(drop = True)

    # save file
    if save_file == True:

        # save as csv
        csv_path = 'data/target/raw'
        os.makedirs(csv_path, exist_ok = True)
        df.to_csv(f'{csv_path}/tested_positive_num_raw.csv', index = False)


# In[ ]:


def get_total_case(start_date,end_date):
    start = datetime.strptime(str(start_date),"%Y.%m.%d")
    end = datetime.strptime(str(end_date),"%Y.%m.%d")

    new_start_date = start.strftime("%Y-%m-%d")
    new_end_date = end.strftime("%Y-%m-%d")

    idx = pd.date_range(f'{new_start_date}',f'{new_end_date}',)


    csv_date = []
    csv_local = []
    csv_total = []

    
    #例外
    patterna = r'((\d+),(\d+))|(\d+)例[\u4e00-\u9fa5]+'

    ##方法一
    for page in range(1,1500):
        url = f'https://www.cdc.gov.tw/Bulletin/List/MmgtpeidAR5Ooai4-fgHzQ?page={page}&startTime={start_date}&endTime={end_date}'
        htmlfile = requests.get(url)
        objSoup = bs4.BeautifulSoup(htmlfile.text, 'lxml')

        objTag = objSoup.select('.cbp-item')    #找到.cbp-item這個class
        #print("objTag串列長度 = ", len(objTag))

        if len(objTag) != 0:    #若selecet回傳的串列不為空則繼續循環(因為不知道會有多少分頁，所以預設了1500頁，如果回傳為空就代表沒有那頁，就結束循環)
            for i in range(len(objTag)):
                covid = objTag[i].find_all('p', {'class':'JQdotdotdot'})    #新聞標題（本日新增xx例本土，xx例境外）
                covid_year = objTag[i].find_all('p',{'class':'icon-year'})  #新聞的年月（2022 - 1）
                covid_date = objTag[i].find_all('p',{'class':'icon-date'})  #新聞的日（12）

                for j in range(len(covid)):
                    covid_text = str(covid[j].text) #將新聞標題文字轉為string
                    pattern = r'新增(\d+)((\S+)?)'
                    result = re.search(pattern, covid_text)
                    real_date = str(covid_year[j].text + '-' + covid_date[j].text).replace(' - ','-')   #整理日期格式
                    if 'print_date' not in locals() or print_date != real_date:
                        print_date = real_date
                        print('getting target data of ', print_date)

                    if result != None:  #如果有案例
                        if ("登革熱" not in covid_text):
                            # 如果兩個都有
                            if ('本土' in covid_text) and ('境外' in covid_text):
                                csv_date.append(real_date)
                                #本土pattern
                                pattern1 = r'(((\d+),(\d+))|(\d+))例((COVID-19)?)本土'
                                #境外pattern
                                pattern2 = r'(((\d+),(\d+))|(\d+))例((COVID-19)?)境外'
                                result1 = re.search(pattern1, covid_text)
                                result2 = re.search(pattern2, covid_text)
                                local_num = int(str(result1.group()).replace('COVID-19','').replace('本土','').replace('例','').replace(',',''))
                                outside_num = int(str(result2.group()).replace('COVID-19','').replace('境外','').replace('例','').replace(',',''))
                                # print(local_num)
                                # print(outside_num)
                                total = local_num + outside_num
                                # print(total)
                                csv_total.append(total)
                            # 只有本土
                            elif ('本土'in covid_text) and ('境外' not in covid_text):
                                csv_date.append(real_date)
                                patterna = r'((\d+),(\d+))|(\d+)((例)?)'
                                result3 = re.search(patterna, covid_text)
                                local_num = int(str(result3.group()).replace('COVID-19','').replace('本土','').replace('例','').replace(',',''))
                                # print(local_num)
                                total = local_num
                                # print(total)
                                csv_total.append(total)
                            # 只有境外
                            elif ('本土' not in covid_text) and ('境外' in covid_text):
                                csv_date.append(real_date)
                                patterna = r'((\d+),(\d+))|(\d+)((例)?)'
                                result4 = re.search(patterna, covid_text)
                                outside_num = int(str(result4.group()).replace('COVID-19','').replace('境外','').replace('例','').replace(',',''))
                                # print(outside_num)
                                total = outside_num
                                # print(total)
                                csv_total.append(total)
                        else:
                            continue
                    else: 
                        continue
        else:
            break

    # print(len(csv_date))
    # print(len(csv_local))
    data = [csv_date, csv_total]
    col = ['日期','當日總確診人數']
    df = pd.DataFrame(list(zip(*data)),columns=col)
    df_new1 = df.groupby(['日期'], sort=False)['當日總確診人數'].sum().reset_index()  #如果同一天有兩條新增本土案例的新聞，則相加
    df_new2 = df_new1.set_index(pd.to_datetime(df_new1['日期']))
    #print(df_new2)

    df_new = df_new2.reindex(idx, fill_value=0)
    df_final = df_new['當日總確診人數']
    
    # add by sheena
    df_final = df_final.to_frame(name = 'tested_positive_num')    
    df_final = df_final.reset_index(level=0)
    df_final = df_final.rename(columns = {'index':'survey_date'})
    
    # print(df_final)
    # df_final.to_csv(f'{new_start_date}_{new_end_date}當日總確診人數.csv')
        
    return df_final


# In[ ]:


def get_domestic_case(start_date, end_date):
    start = datetime.strptime(str(start_date),"%Y.%m.%d")
    end = datetime.strptime(str(end_date),"%Y.%m.%d")

    new_start_date = start.strftime("%Y-%m-%d")
    new_end_date = end.strftime("%Y-%m-%d")

    idx = pd.date_range(f'{new_start_date}',f'{new_end_date}',)


    csv_date = []
    csv_local = []

    ##方法一
    for page in range(1,1500):
        url = f'https://www.cdc.gov.tw/Bulletin/List/MmgtpeidAR5Ooai4-fgHzQ?page={page}&startTime={start_date}&endTime={end_date}'
        htmlfile = requests.get(url)
        objSoup = bs4.BeautifulSoup(htmlfile.text, 'lxml')

        objTag = objSoup.select('.cbp-item')    #找到.cbp-item這個class
        #print("objTag串列長度 = ", len(objTag))

        if len(objTag) != 0:    #若selecet回傳的串列不為空則繼續循環(因為不知道會有多少分頁，所以預設了1500頁，如果回傳為空就代表沒有那頁，就結束循環)
            for i in range(len(objTag)):
                covid = objTag[i].find_all('p', {'class':'JQdotdotdot'})    #新聞標題（本日新增xx例本土，xx例境外）
                covid_year = objTag[i].find_all('p',{'class':'icon-year'})  #新聞的年月（2022 - 1）
                covid_date = objTag[i].find_all('p',{'class':'icon-date'})  #新聞的日（12）

                for j in range(len(covid)):
                    covid_text = str(covid[j].text) #將新聞標題文字轉為string
                    #找出本土的例子
                    pattern = r'(\d+)例((COVID-19)?)本土'
                    result = re.search(pattern, covid_text)
                    real_date = str(covid_year[j].text + '-' + covid_date[j].text).replace(' - ','-')   #整理日期格式
                    if 'print_date' not in locals() or print_date != real_date:
                        print_date = real_date
                        print('getting target data of ', print_date)

                    if result != None:  #如果有本土案例
                        if ('新增' in covid_text) and ('例本土' in covid_text) and ('境外' in covid_text):
                            csv_date.append(real_date)
                            pattern1 = r'(((\d+),(\d+))|(\d+))例((COVID-19)?)本土'
                            result1 = re.search(pattern1, covid_text)
                            local_num = str(result1.group().replace(r'例本土','')).replace(',','')
                            # print(local_num)
                            csv_local.append(int(local_num))
                        elif ('新增' in covid_text) and ('例COVID-19本土' in covid_text) and ('境外' in covid_text):
                            csv_date.append(real_date)
                            pattern2 = r'((\d+),(\d+))|(\d+)例COVID-19本土'
                            result2 = re.search(pattern2, covid_text)
                            local_num = str(result2.group().replace(r'例COVID-19本土','')).replace(',','')
                            # print(local_num)
                            csv_local.append(int(local_num))
                        elif ('新增' in covid_text) and ('本土' in covid_text) and ('境外' not in covid_text):
                            csv_date.append(real_date)
                            pattern3 = r'((\d+),(\d+))|(\d+)例'
                            result3 = re.search(pattern3, covid_text)
                            local_num = str(result3.group().replace(r'例','')).replace(',','')
                            # print(local_num)
                            csv_local.append(int(local_num))
                    else: 
                        continue
        else:
            break

    # print(len(csv_date))
    # print(len(csv_local))
    data = [csv_date, csv_local]
    col = ['日期','本土確診人數']
    df = pd.DataFrame(list(zip(*data)),columns=col)
    df_new1 = df.groupby(['日期'], sort=False)['本土確診人數'].sum().reset_index()  #如果同一天有兩條新增本土案例的新聞，則相加
    df_new2 = df_new1.set_index(pd.to_datetime(df_new1['日期']))
    #print(df_new2)

    df_new = df_new2.reindex(idx, fill_value=0)
    df_final = df_new['本土確診人數']
    
    # add by sheena
    df_final = df_final.to_frame(name = 'tested_positive_num')    
    df_final = df_final.reset_index(level=0)
    df_final = df_final.rename(columns = {'index':'survey_date'})
    
    # print(df_final)
    # df_final.to_csv(f'{new_start_date}_{new_end_date}本土確診人數.csv')
    
    return df_final


# In[ ]:


def read_tested_positive(date_start, date_end, save_file = True):
    
    # read data
    if os.path.exists(f'./data/target/raw/tested_positive_num_raw.csv'):
        df = pd.read_csv(f'./data/target/raw/tested_positive_num_raw.csv')
        df["survey_date"] = pd.to_datetime(df["survey_date"], format = "%Y-%m-%d")
    
    # produce time to start and end
    if os.path.exists(f'./data/target/raw/tested_positive_num_raw.csv'):
        start = df["survey_date"].tolist()[-1] + timedelta(days=1)
        end = datetime.strptime(date_end, '%Y%m%d')
    else:
        start = datetime.strptime(date_start, '%Y%m%d')
        end = datetime.strptime(date_end, '%Y%m%d')
    
    # check if government has report today
    report_time_str = '15:00'
    report_time = datetime.strptime(report_time_str, '%H:%M').time()
    report_time_today = datetime.combine(date.today(), report_time)
    today = datetime.combine(date.today(), datetime.min.time())
    if end == today:
        is_today = True
    if is_today and datetime.now() < report_time_today:
        end -= timedelta(days=1)
    
    # add data of current date
    start_date = start.strftime("%Y.%m.%d")
    end_date = end.strftime("%Y.%m.%d")
    if pd.Timestamp(start) <= pd.Timestamp(end):
        new_data = get_total_case(start_date, end_date)
        if os.path.exists(f'./data/target/raw/tested_positive_num_raw.csv'):
            df = df.append(new_data, ignore_index = True)
            if save_file == True:
                df.to_csv(f'./data/target/raw/tested_positive_num_raw.csv', index = False)
        else:
            df = new_data.copy()
            if save_file == True:
                csv_path = 'data/target/raw'
                os.makedirs(csv_path, exist_ok = True)
                df.to_csv(f'{csv_path}/tested_positive_num_raw.csv', index = False)
    
    # append future data
    if is_today:
        fut = df[-1:].copy()
        fut['survey_date'] = fut.survey_date + timedelta(days=1)
        df = df.append(fut, ignore_index = True)
    
    return df


# In[ ]:


def read_tested_positive_local(date_start, date_end, save_file = True):
    
    # read data
    if os.path.exists(f'./data/target/raw/tested_positive_num_local_raw.csv'):
        df = pd.read_csv(f'./data/target/raw/tested_positive_num_local_raw.csv')
        df["survey_date"] = pd.to_datetime(df["survey_date"], format = "%Y-%m-%d")
    
    # produce time to start and end
    if os.path.exists(f'./data/target/raw/tested_positive_num_local_raw.csv'):
        start = df["survey_date"].tolist()[-1] + timedelta(days=1)
        end = datetime.strptime(date_end, '%Y%m%d')
    else:
        start = datetime.strptime(date_start, '%Y%m%d')
        end = datetime.strptime(date_end, '%Y%m%d')
    
    # check if government has report today
    report_time_str = '15:00'
    report_time = datetime.strptime(report_time_str, '%H:%M').time()
    report_time_today = datetime.combine(date.today(), report_time)
    today = datetime.combine(date.today(), datetime.min.time())
    if end == today:
        is_today = True
    if is_today and datetime.now() < report_time_today:
        end -= timedelta(days=1)
    
    # add data of current date
    start_date = start.strftime("%Y.%m.%d")
    end_date = end.strftime("%Y.%m.%d")
    if pd.Timestamp(start) <= pd.Timestamp(end):
        new_data = get_domestic_case(start_date, end_date)
        if os.path.exists(f'./data/target/raw/tested_positive_num_local_raw.csv'):
            df = df.append(new_data, ignore_index = True)
            if save_file == True:
                df.to_csv(f'./data/target/raw/tested_positive_num_local_raw.csv', index = False)
        else:
            df = new_data.copy()
            if save_file == True:
                csv_path = 'data/target/raw'
                os.makedirs(csv_path, exist_ok = True)
                df.to_csv(f'{csv_path}/tested_positive_num_local_raw.csv', index = False)
    
    # append future data
    if is_today:
        fut = df[-1:].copy()
        fut['survey_date'] = fut.survey_date + timedelta(days=1)
        df = df.append(fut, ignore_index = True)
    
    return df


# In[ ]:


def confirmed_cases():

    response = requests.get("https://covid-19.nchc.org.tw/api/covid19?CK=covid-19@nchc.org.tw&querydata=4001&limited=TWN").text

    # transfer json to pandas
    jsonData = json.loads(response)
    df = pd.DataFrame.from_dict(jsonData)

    # rename columns
    df.rename(columns = {"id":"ID","a01":"iso_code","a02":"洲名","a03":"國家","a04":"日期","a05":"總確診數","a06":"新增確診數","a07":"七天移動平均新增確診數","a08":"總死亡數","a09":"新增死亡數","a10":"七天移動平均新增死亡數","a11":"每百萬人確診數","a12":"每百萬人死亡數","a13":"傳染率","a14":"新增檢驗件數","a15":"總檢驗件數","a16":"每千人檢驗件數","a17":"七天移動平均新增檢驗件數","a18":"陽性率","a19":"每確診案例相對檢驗數量","a20":"疫苗總接種總劑數","a21":"疫苗總接種人數","a22":"疫苗新增接種劑數","a23":"七天移動平均疫苗新增接種劑數","a24":"每百人接種疫苗劑數","a25":"每百人接種疫苗人數","a26":"疫情控管指數","a27":"總人口數","a28":"中位數年紀","a29":"70歲以上人口比例","a30":"平均壽命","a31":"解除隔離數","a32":"解封指數"}, inplace = True)

    # keep useable columns and rename into English
    df = df[["日期", "傳染率", "疫苗總接種總劑數", "疫苗總接種人數", "解封指數"]]
    df = df.rename(columns = {"日期": "survey_date", 
                              "傳染率": "reproduction_rate", 
                              "疫苗總接種總劑數": "total_vaccinations", 
                              "疫苗總接種人數": "people_vaccinated", 
                              "解封指數": "unblock"})

    # sort by survey date
    df["survey_date"] = pd.to_datetime(df["survey_date"], format = "%Y-%m-%d")
    df = df.sort_values(by = "survey_date")
    df = df.reset_index(drop = True)

    # process total_vaccinations
    total1 = 0
    t_vac = df['total_vaccinations'].tolist()
    t_vac = list(map(int, t_vac))

    for i in range(len(t_vac)):
        if t_vac[i] > total1:
            total1 = t_vac[i]
        elif t_vac[i] < total1:
            t_vac[i] = total1

    df['total_vaccinations'] = pd.DataFrame(t_vac)

    # process people_vaccinated
    total2 = 0
    p_vac = df['people_vaccinated'].tolist()
    p_vac = list(map(int, p_vac))

    for i in range(len(p_vac)):
        if p_vac[i] > total2:
            total2 = p_vac[i]
        elif p_vac[i] < total2:
            p_vac[i] = total2

    df['people_vaccinated'] = pd.DataFrame(p_vac)

    return df


# In[ ]:


def stats():

    response = requests.get("https://covid-19.nchc.org.tw/api/covid19?CK=covid-19@nchc.org.tw&querydata=4003").text

    # transfer json to pandas
    jsonData = json.loads(response)
    df = pd.DataFrame.from_dict(jsonData)

    # rename columns
    df.rename(columns = {"id":"ID","a01":"日期","a02":"法定傳染病通報","a03":"居家檢疫送驗","a04":"擴大監測送驗","a05":"送驗(今日合計)","a06":"送驗(總累計)","a07":"排除(總累計)","a08":"昨日排除","a09":"昨日送驗"}, inplace = True)

    # keep useable columns and rename into English
    df = df[["日期", "法定傳染病通報", "居家檢疫送驗", "擴大監測送驗"]]
    df = df.rename(columns = {"日期": "survey_date", 
                              "法定傳染病通報": "enhanced_surveillance", 
                              "居家檢疫送驗": "reported_from_home_quarantine", 
                              "擴大監測送驗": "reported_covid19_cases"})

    # sort by survey date
    df["survey_date"] = pd.to_datetime(df["survey_date"], format = "%Y-%m-%d")
    df = df.sort_values(by = "survey_date")
    df = df.reset_index(drop = True)
    
    return df


# In[ ]:


def get_official_data(save_file = True):
    
    df1 = confirmed_cases()
    df2 = stats()

    df = pd.merge(df1, df2, on=['survey_date'])

    # arrange data
    ## for reproduction_rate, new_tests, total_vaccinations, people_vaccinated, unblock, enhanced_surveillance,
    ## reported_from_home_quarantine and reported_covid19_cases, we will use the data of the day before yesterday,
    ## since we cannot get these information timely
    df["survey_date"] = pd.to_datetime(df["survey_date"], format = "%Y-%m-%d")
    f1 = df[-1:].copy()
    f2 = df[-1:].copy()
    f1['survey_date'] = f1.survey_date + timedelta(days=1)
    f2['survey_date'] = f2.survey_date + timedelta(days=2)
    df = df.append(f1, ignore_index = True)
    df = df.append(f2, ignore_index = True)
    
    arr = ['reproduction_rate', 'total_vaccinations', 'people_vaccinated', 'unblock', 
           'enhanced_surveillance', 'reported_from_home_quarantine', 'reported_covid19_cases']

    for i in range(len(arr)):
        df[f'{arr[i]}_pre'] = df[arr[i]].shift(2)
        df = df.drop(columns = [arr[i]])
    
    df = df.dropna()
    
    # save file
    if save_file == True:

        # save as csv
        csv_path = 'data/official'
        os.makedirs(csv_path, exist_ok = True)        
        df.to_csv(f'{csv_path}/official.csv', index = False)


# In[ ]:


def get_UMD_data(date_start, date_end, dtype = 'smoothed', save_file = True):
    
    # get feature list
    feat_df = pd.read_csv('../features.csv')
    feat = feat_df['features'].tolist()
    
    data_ext = []
    data_err = []
    data_else = []
    
    for i in range(len(feat)):

        # request data from api
        response = requests.get(f"https://covidmap.umd.edu/api/resources?indicator={feat[i]}&type={dtype}&country=Taiwan&daterange={date_start}-{date_end}").text

        # convert json data to dic data for use!
        jsonData = json.loads(response)

        # convert to pandas dataframe
        if 'data' in jsonData:     # if data exists
            df = pd.DataFrame.from_dict(jsonData['data'])
            if df.empty:
                data_else.append(feat[i])     # check if df is empty
            else:
                data_ext.append(feat[i])
                
                # save file
                if save_file == True:
                    csv_path = f'data/UMD/{dtype}'
                    os.makedirs(csv_path, exist_ok = True)
                    df.to_csv(f'{csv_path}/{feat[i]}.csv', index = False)

            # print(feat[i], ':\n', df, '\n')

        elif 'error' in jsonData:     # if error exists
            data_err.append(feat[i])
            # print(feat[i], ': error\n')

        else:
            data_else.append(feat[i])
            # print(feat[i], ': else\n')
    
    # print('How many data exists: ', len(data_ext))
    # print('How many data error: ', len(data_err))
    # print('How many data left: ', len(data_else))
    # print('Data exists: ', data_ext)
    # print('Data error: ', data_err)
    # print('Data left: ', data_else)


# In[ ]:


def get_oxcgrt_data(date_start, date_end, save_file = True):

    # convert input date to datetime object
    start = datetime.strptime(str(date_start),"%Y%m%d")
    end = datetime.strptime(str(date_end),"%Y%m%d")

    # set time interval
    day = timedelta(days=1)
    mydate = start
    date_list = []
    while mydate <= end:
        newdate = mydate.strftime("%Y-%m-%d")
        date_list.append(newdate)
        mydate += day

    
    # fetch data from official website (from date_start to date_end)
    response = requests.get(f"https://covidtrackerapi.bsg.ox.ac.uk/api/v2/stringency/date-range/{date_start}/{date_end}").text
    jsonData = json.loads(response)
    record = pd.json_normalize(jsonData["data"], errors='ignore')
    oxcgrt = []
    

    for i in date_list:
        x = i + ".TWN" + ".stringency_actual"
        if x in record.columns :
            oxcgrt.append(record[x].values[0])
        else:
            oxcgrt.append(0)
            
    df = pd.DataFrame({'survey_date':date_list,'stringency_actual':oxcgrt})

    # save file
    if save_file == True:

        # save as csv
        csv_path = 'data/OxCGRT'
        os.makedirs(csv_path, exist_ok = True)        
        df.to_csv(f'{csv_path}/oxcgrt_day.csv', index = False, na_rep = 0)


# In[ ]:


def smooth_data(df, date_start, date_end, target, mv_day=[1,7,14], save_file = True, scale = True, target_or_not = True):
    
    date = df['survey_date'].tolist()
    data = df[target].tolist()

    for i in range(len(mv_day)):

        day_num = mv_day[i]
        mv_date = date[day_num-1:]
        mv_data = []
        
        # calculate moving average
        for j in range(len(data)-day_num+1):
            mv_data_value = np.mean(list(map(int, data[j:j+day_num])))
            mv_data.append(mv_data_value)
        
        # scale value
        if scale == True:
            if max(mv_data) >= 1:
                d = int(math.log10(max(mv_data)))+1
                mv_data = [i * (0.1 ** d) for i in mv_data]
                mv_data = [round(i, d) for i in mv_data]
        
        # build dataframe
        mv_df = pd.DataFrame({'survey_date': mv_date, f'{target}_smoothed_{day_num}d': mv_data})
        
        # drop time out of range
        start = datetime.strptime(date_start, '%Y%m%d')
        end = datetime.strptime(date_end, '%Y%m%d')
        mv_df = mv_df.drop(mv_df[(mv_df.survey_date<start)].index)
        mv_df = mv_df.drop(mv_df[(mv_df.survey_date>end)].index)
        
        # save file
        locals()[f'data_smoothed_{day_num}d'] = mv_df
        if save_file == True:
            csv_path = 'data/target/smoothed' if target_or_not == True else 'data/smoothed'
            os.makedirs(csv_path, exist_ok = True)
            locals()[f'data_smoothed_{day_num}d'].to_csv(f'{csv_path}/{target}_smoothed_{day_num}d.csv', index = False)


# In[ ]:


def get_data(date_start, date_end, mv_day, target_type = 'all', smoothed=True, get_official=True, get_oxcgrt=True, get_UMD=True):
    
    # get target data
    # get_tested_positive()
    target_data = read_tested_positive_local(date_start, date_end) if target_type == 'local' else read_tested_positive(date_start, date_end)
    if smoothed == True:
        smooth_data(target_data, date_start, date_end, 'tested_positive_num', mv_day = mv_day, scale = False)
    
    # get official data
    if get_official:
        get_official_data()
    
    # get OxCGRT data
    if get_oxcgrt:
        get_oxcgrt_data(date_start, date_end)
    
    # get UMD data
    if get_UMD:
        get_UMD_data(date_start, date_end)
        
    print('Finished getting data.')


# In[ ]:




