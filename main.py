import pandas as pd
from datetime import date, datetime, timedelta

import function as func

def get_today():
    today = date.today()
    d = today.strftime('%Y%m%d')
    return d

def prod_date_end(test_end, test_num):
    last = datetime.strptime(test_end, '%Y%m%d')
    end = last - timedelta(days=test_num-1)
    date_end = end.strftime("%Y%m%d")
    return date_end

def prod_train_num(train_start, train_end):
    start = datetime.strptime(train_start, '%Y%m%d')
    end = datetime.strptime(train_end, '%Y%m%d')
    train_num = (end-start).days
    return train_num

def get_output(df, day_num, target_type = 'total'):
    
    # check if day_num = 1 is reasonable
    day_num = day_num.copy()
    if get_official == False and get_oxcgrt == False and get_UMD == False:
        day_num.remove(1)
    
    # read predicted number
    case_predict = []
    for day in day_num:
        pred_list = df[f'preds_{day}'].tolist()
        case_predict.append(pred_list[0])
    
    # print
    target = '本土' if target_type == 'local' else '總數'
    print(f'\n*{target}*')
    for i in range(len(day_num)):
        print(f'使用前{day_num[i]}天數據預測： %.2f 例' % case_predict[i])
    
    return case_predict


'''Set up'''

date_start = '20200101'
date_end = get_today()
mv_day = [1]
train_start = '20200116'
train_end = date_end
# train_num = prod_train_num(train_start, train_end)
train_num = 20
data_num_thres = 100
corr_thres = 0.3
day_num = [1,3,7,10,14] # must have 1

get_official = True
get_oxcgrt = False
get_UMD = False

print_process = True
get_data_or_not = True
calc_corr_or_not = True
data_sort_or_not = True

# model parameters
target_only = False

config = {
    'model_num': 2,                         # model type
    'n_epochs': 5000,                       # maximum number of epochs
    'batch_size': 4,                        # mini-batch size for dataloader (org: 16)
    'optimizer': 'Adam',                    # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {                       # hyper-parameters for the optimizer
        'lr': 0.001,                        # learning rate
        'weight_decay': 0.0005,             # weight decay: to avoid overfitting
        'betas': (0.9, 0.999)
        # 'momentum': 0.9                     # momentum for SGD
    },
    'early_stop': 50,                       # early stopping epochs (the number epochs since model's last improvement) (org:350)
    'save_path': './models/model.pth'       # save model
}


'''Training for local case prediction'''

print('*Start predicting local cases*\n')

target_type = 'local'
exp_num = 'local'

# data preprocess
func.get_data(date_start, date_end, mv_day, target_type, get_official=get_official, get_oxcgrt=get_oxcgrt, get_UMD=get_UMD) if get_data_or_not == True else print('Not getting data.')
func.calc_corr(train_start, train_end, train_num, mv_day, data_num_thres, corr_thres, get_UMD=get_UMD) if calc_corr_or_not == True else print('Not calculating correlation.')
target_range = func.data_sort(date_start, date_end, train_num, mv_day, day_num, get_official=get_official, get_oxcgrt=get_oxcgrt, get_UMD=get_UMD) if data_sort_or_not == True else print('Not sorting data.')

# prediction
func.prediction_DNN(exp_num, day_num, mv_day, target_only, config, target_range, print_process)


'''Training for total case prediction'''

print('*Start predicting total cases*\n')

# data preprocess
func.get_data(date_start, date_end, mv_day, target_type, get_official=get_official, get_oxcgrt=get_oxcgrt, get_UMD=get_UMD) if get_data_or_not == True else print('Not getting data.')
func.calc_corr(train_start, train_end, train_num, mv_day, data_num_thres, corr_thres, get_UMD=get_UMD) if calc_corr_or_not == True else print('Not calculating correlation.')
target_range = func.data_sort(date_start, date_end, train_num, mv_day, day_num, get_official=get_official, get_oxcgrt=get_oxcgrt, get_UMD=get_UMD) if data_sort_or_not == True else print('Not sorting data.')

# prediction
func.prediction_DNN(exp_num, day_num, mv_day, target_only, config, target_range, print_process)


'''Produce output'''

df_local = pd.read_csv('./results/data_prediction/data_pred_exp_local_num_smoothed_1d.csv')
df_total = pd.read_csv('./results/data_prediction/data_pred_exp_total_num_smoothed_1d.csv')

print(f'{date_end}確診人數預測值')
predict_local = get_output(df_local, day_num, target_type = 'local')
predict_total = get_output(df_total, day_num, target_type = 'total')

# Predicted local cases are saved in list "predict_local"
# while predicted total cases are saved in list "predict_total"