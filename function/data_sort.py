#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import random
import datetime
from datetime import datetime, timedelta
from .calculate_correlation import prepare_date_range
pd.options.mode.chained_assignment = None


# In[ ]:


def produce_output_name(target_name):
    output_name = target_name.replace('tested_positive_', "")
    output_name = output_name.replace('_', '.')
    return output_name


# In[ ]:


def get_feature_list(target_name, corr_start, corr_end):
    df_corr = pd.read_csv(f'./correlation/correlation_{target_name}_{corr_start}_{corr_end}.csv')
    feat_list = df_corr['features'].tolist()
    return feat_list


# In[ ]:


def normalize_data(df):
    
    for col in df.columns[2:]:
        min_df = df[col].min()
        max_df = df[col].max()
        for i in range(len(df)):
            df[col].iloc[i] = (df[col].iloc[i]-min_df)/(max_df-min_df)
            
    return df


# In[ ]:


def merge_data(target_name, corr_start, corr_end, get_official=True, get_oxcgrt=True, get_UMD=True):
    
    # blank dataframe
    data = pd.DataFrame()
    
    # merge official data
    if get_official:
        dt = pd.read_csv('./data/official/official.csv')
        dt['survey_date'] = pd.to_datetime(dt['survey_date'], format="%Y-%m-%d")
        data = dt if data.empty == True else pd.merge(data, dt, on=['survey_date'])

    # merge oxcgrt data
    if get_oxcgrt:
        dt = pd.read_csv('./data/OxCGRT/oxcgrt_day.csv')
        dt['survey_date'] = pd.to_datetime(dt['survey_date'], format="%Y-%m-%d")
        data = dt if data.empty == True else pd.merge(data, dt, on=['survey_date'])
    
    # merge UMD data
    if get_UMD:
        feat_list = get_feature_list(target_name, corr_start, corr_end)
        for i in range(len(feat_list)):
            data_name = feat_list[i]
            dt = pd.read_csv(f'./data/UMD/smoothed/{data_name}.csv')                    ## read data
            dt.columns = [data_name] + list(dt.columns[1:])                              ## rename needed columns
            dt['survey_date'] = pd.to_datetime(dt['survey_date'], format="%Y%m%d")       ## convert survey_date datatype to datatime
            dt = dt[['survey_date', data_name]]                                          ## drop useless columns and rearrange
            data = dt if data.empty == True else pd.merge(data, dt, on=['survey_date'])  ## merge data into dataframe

    # merge target data
    dt = pd.read_csv(f'./data/target/smoothed/{target_name}.csv')
    dt['survey_date'] = pd.to_datetime(dt['survey_date'], format="%Y-%m-%d")     ## convert survey_date datatype to datatime
    data = dt if data.empty == True else pd.merge(data, dt, on=['survey_date'])  ## merge data into dataframe

    
    target_range = [data[target_name].min(), data[target_name].max()]  # get target range
    data = normalize_data(data)
    
    return data, target_range


# In[ ]:


def save_data(data, data_final, day, output_name, train_num=20):
    
    # Produce index to drop
    train_list = list(range(0, len(data_final)-1))
    test_list = list(range(len(data_final)-1, len(data_final)))

    # covid.train
    train = data_final
    drop_index = test_list
    drop = train.drop(drop_index, inplace = False)   # delete rows out of boundary
    drop.index = range(len(drop))
    drop.reset_index(drop=True, inplace=True)   # reindex
    train_time = drop
    train_notime = drop.drop(columns = ['survey_date'])

    # covid.test
    test = data_final
    drop_index = train_list
    drop = test.drop(drop_index, inplace = False)   # delete rows out of boundary
    drop.index = range(len(drop))
    drop.reset_index(drop=True, inplace=True)   # reindex
    test_time = drop
    test_notime = drop.drop(columns = ['survey_date'])

    # download training data
    csv_path_training = './data/training'
    csv_path_withdate = './data/training/withdate'
    csv_path_all = './data/all'
    os.makedirs(csv_path_training, exist_ok = True)
    os.makedirs(csv_path_withdate, exist_ok = True)
    os.makedirs(csv_path_all, exist_ok = True)
    
    train_notime.to_csv(f'{csv_path_training}/covid.train.{day}day.{output_name}.csv')
    test_notime.to_csv(f'{csv_path_training}/covid.test.{day}day.{output_name}.csv')
    train_time.to_csv(f'{csv_path_withdate}/covid.train.{day}day.withdate.{output_name}.csv')
    test_time.to_csv(f'{csv_path_withdate}/covid.test.{day}day.withdate.{output_name}.csv')
    data_final.to_csv(f'{csv_path_all}/covid.{day}day.{output_name}.csv')
    data.to_csv(f'{csv_path_all}/all.{output_name}.csv')


# In[ ]:


def build_df_of_day_num(data, day_num, output_name, train_num=20):
    
    # build blank dataframe
    data_final = pd.DataFrame()

    for k in range(len(day_num)):

        '''Seperate days'''
        for i in range(day_num[k]):

            # build data_day
            data_day = data

            # add suffix to columns
            data_day = data_day.add_suffix(f'_{i+1}')

            # rename survey_date and shifted_date
            data_day = data_day.rename(columns={f'survey_date_{i+1}': 'survey_date'})

            # shift datetime
            data_day['survey_date'] = data_day.survey_date + timedelta(days=day_num[k]-i-1)

            # name data_day
            globals()[f'data_{i+1}'] = data_day
        
        '''Merge data_day'''
        # build data_final
        data_final = data_1

        for i in range(day_num[k]-1):
            data_day = globals()[f'data_{i+2}']
            data_final = pd.merge(data_final, data_day, on=['survey_date'])    # merge by shifted date

        # build dataframe with survey_date
        data_final_time = data_final
        globals()[f'data_{day_num[k]}_day_time'] = data_final_time

        # build dataframe without survey_date
        data_final = data_final.drop(columns = ['survey_date'])

        '''Save data'''
        save_data(data, data_final_time, day_num[k], output_name, train_num)


# In[ ]:


def data_sort(date_start, date_end, train_num, mv_day, day_num, get_official=True, get_oxcgrt=True, get_UMD=True):
    
    # corr_start, corr_end = prepare_date_range(date_start, train_num)
    corr_start, corr_end = date_start, date_end
    
    for i in range(len(mv_day)):
        target_name = f'tested_positive_num_smoothed_{mv_day[i]}d'        
        output_name = produce_output_name(target_name)
        data, target_range = merge_data(target_name, corr_start, corr_end, get_official=get_official, get_oxcgrt=get_oxcgrt, get_UMD=get_UMD)
        build_df_of_day_num(data, day_num, output_name, train_num)
    
    print('Finished sorting data.')
    
    return target_range


# In[ ]:




