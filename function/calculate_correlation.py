#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import openpyxl
import os
from datetime import datetime, timedelta


# In[ ]:


def prepare_date_range(date_start, train_num):
    
    start = datetime.strptime(date_start, "%Y%m%d")
    end = start + timedelta(days=train_num-1)
    
    date_end = end.strftime("%Y%m%d")
    
    return date_start, date_end


# In[ ]:


def prepare_target(target_name):

    # read target
    tar_path = f'data/target/smoothed/{target_name}.csv'  ## path to target
    target = pd.read_csv (tar_path)

    # rearrange columns
    target = target[['survey_date', f'{target_name}']]

    # convert survey_date datatype to datatime
    target['survey_date'] = pd.to_datetime(target['survey_date'], format="%Y-%m-%d")

    return target


# In[ ]:


def record_correlation(target, target_name, date_start, date_end):
    
    # get feature list
    feat_list = os.listdir('data/UMD/smoothed')    # read all data
    
    # get date range
    start = datetime.strptime(date_start, "%Y%m%d")
    end = datetime.strptime(date_end, "%Y%m%d")
    
    # calculate correlation between features and target
    col_name = []
    col_cor = []
    col_len = []
    col_start = []
    col_end = []
    
    for i in range(len(feat_list)):

        # read data
        feat_path = f'data/UMD/smoothed/{feat_list[i]}'     ## path to features
        data_name = feat_list[i].replace(".csv", "")        ## deal with data name
        data = pd.read_csv (feat_path)                      ## read csv

        # deal with data
        data.columns = [data_name] + list(data.columns[1:])                           ## rename needed columns
        data = data[['survey_date', data_name]]                                       ## drop useless columns and rearrange
        data['survey_date'] = pd.to_datetime(data['survey_date'], format="%Y%m%d")    ## convert survey_date datatype to datatime

        # merge with target data
        comp = pd.merge(data, target, on=['survey_date'])
        comp = comp.drop(comp[(comp.survey_date<start)].index)

        # record overlap length
        overlap_len = len(comp)
        if overlap_len > 2:
            data_start = comp.iloc[0].iat[0]
            data_end = comp.iloc[-1].iat[0]
        else:
            data_start = float('nan')
            data_end = float('nan')

        # delete data out of date range
        comp = comp.drop(comp[(comp.survey_date>end)].index)

        # compare with target data
        data_len = len(comp)
        if data_len > 2:
            cor_mat = np.corrcoef(comp[[data_name, f'{target_name}']].values, rowvar = False, ddof=0)
            data_cor = cor_mat[0][1]  
        else:
            data_cor = float('nan') 

        # record results
        col_name.append(data_name)
        col_cor.append(data_cor)
        col_len.append(overlap_len)

        # record time range
        col_start.append(data_start)
        col_end.append(data_end)
    
    # build dataframe
    df_cor = pd.DataFrame({'features': col_name, 'correlation': col_cor, 'data_num': col_len,
                           'start_time': col_start, 'end_time': col_end})
    
    # save file
    csv_path = 'correlation'
    os.makedirs(csv_path, exist_ok = True)
    output_name = target_name.replace('tested_positive_', "")
    df_cor.to_csv(f'{csv_path}/correlation_{target_name}_{date_start}_{date_end}_all.csv', index = False)
    
    return df_cor


# In[ ]:


def under_condition(df, data_num_thres, corr_thres, target_name, date_start, date_end):
    
    col_name = df['features'].tolist()
    col_cor = df['correlation'].tolist()
    col_len = df['data_num'].tolist()
    
    # produce index to drop
    pop_list = []
    for i in range(len(col_name)):
        if (col_len[i] < data_num_thres) or ((col_cor[i] < corr_thres) and (col_cor[i] > -corr_thres) or np.isnan(col_cor[i]) == True):
            pop_list.append(i)

    # delete data
    df_cor_cond = df.drop(pop_list, inplace = False)
    df_cor_cond.reset_index(drop=True, inplace=True)   # reindex
    
    # save file
    csv_path = 'correlation'
    os.makedirs(csv_path, exist_ok = True)
    output_name = target_name.replace('tested_positive_', "")
    df_cor_cond.to_csv(f'{csv_path}/correlation_{target_name}_{date_start}_{date_end}.csv', index = False)
    
    return df_cor_cond


# In[ ]:


def calculate_correlation(date_start, date_end, train_num, mv_day, data_num_thres, corr_thres, get_UMD=True):
    
    # date_start, date_end = prepare_date_range(date_start, train_num)
    if get_UMD:
        for i in range(len(mv_day)):
            target_name = f'tested_positive_num_smoothed_{mv_day[i]}d'        
            target = prepare_target(target_name)
            df_cor = record_correlation(target, target_name, date_start, date_end)
            df_cor_cond = under_condition(df_cor, data_num_thres, corr_thres, target_name, date_start, date_end)
    else:
        pass
    
    print('Finished calculating correlation.')

