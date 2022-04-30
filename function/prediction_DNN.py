#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For data preprocess
import numpy as np
import csv
import os
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


# In[ ]:


def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# In[ ]:


def plot_learning_curve(loss_record, title=''):
    
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 0.5)
    # plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    
    os.makedirs('./results/plot_learning_curve', exist_ok = True)
    plt.savefig(f'./results/plot_learning_curve/learning_curve_day{day}_{exp_name}')
    
    plt.close()
    # plt.show()


# In[ ]:


def plot_valid(preds, targets, lim=1.):

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.02, lim], [-0.02, lim], c='b')
    plt.xlim(-0.02, lim)
    plt.ylim(-0.02, lim)
    # plt.plot([-0.2, lim], [-0.2, lim], c='b')
    # plt.xlim(-0.2, lim)
    # plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Validation')
    
    os.makedirs('./results/plot_validation', exist_ok = True)
    plt.savefig(f'./results/plot_validation/validation_day{day}_{exp_name}')
    
    plt.close()
    # plt.show()


# In[ ]:


def plot_pred(preds, targets, lim=1.):

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.02, lim], [-0.02, lim], c='b')
    plt.xlim(-0.02, lim)
    plt.ylim(-0.02, lim)
    # plt.plot([-0.2, lim], [-0.2, lim], c='b')
    # plt.xlim(-0.2, lim)
    # plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    
    os.makedirs('./results/plot_prediction', exist_ok = True)
    plt.savefig(f'./results/plot_prediction/prediction_day{day}_{exp_name}')
    
    plt.close()
    # plt.show()


# In[ ]:


def plot_predicted_result(preds, targets):
    
    # get x axis
    df = pd.read_csv(f'./data/training/withdate/covid.test.1day.withdate.{read_name}.csv')
    date = df['survey_date'].tolist()
    date = [datetime.strptime(i, "%Y-%m-%d") for i in date]
    date = [date[i] for i in range(len(preds))]
    
    figure(figsize=(9, 6))
    plt.plot(date, preds, label = 'prediction')
    plt.plot(date, targets, label = 'ground truth')
    plt.xticks(np.arange(date[0], date[-1], step=timedelta(days=15)), rotation = 20)
    # plt.yticks(np.arange(0, max(max(preds), max(targets)), step=50))
    plt.xlabel('Date')
    plt.ylabel('Tested Positive Number')
    plt.legend()
    
    os.makedirs('./results/plot_predicted_result', exist_ok = True)
    plt.savefig(f'./results/plot_predicted_result/pred_{exp_name}_day{day}')
    
    plt.close()
    plt.show()


# In[ ]:


def correct_data_scaled(preds, targets):
    
    # find min excluding zero
    a = targets
    n = np.min(a[np.nonzero(a)])
    n = "{:.6f}".format(n).rstrip('0').rstrip('.')
    n = Decimal(str(n))

    # find digits of min
    d = 0
    rem = 1
    while rem != 0:
        n = n * 10
        rem = round(n % 1, 4)
        d += 1
    
    # find multiple
    m = 10 ** d
    
    # correct data
    preds = preds * m
    targets = targets * m    
    preds = [round(i, 3) for i in preds]
    targets = [round(i, 3) for i in targets]
    
    return preds, targets


# In[ ]:


def correct_data_norm(preds, targets, target_range):
    
    tar_min = target_range[0]
    tar_max = target_range[1]
    
    preds = preds * (tar_max-tar_min) + tar_min
    targets = targets * (tar_max-tar_min) + tar_min
    
    return preds, targets


# In[ ]:


class COVID19Dataset(Dataset):
    ''' Dataset for loading and preprocessing the COVID19 dataset '''
    def __init__(self,
                 path,
                 mode='train',
                 target_only=False):
        self.mode = mode

        # Read data into numpy arrays
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)
        
        if not target_only:
            # feats = [:, -1]
            # feats = list(range(day*feat_num-1))
            # feats = list(range(day*feat_num-1))
            feats = list(range(data.shape[1]))
            
        else:
            # TODO: Using 40 states & 2 tested_positive features (indices = 57 & 75)
            # feats = [57,75] + [40,41,42,43] + [58,59,60,61] + [76,77,78,79]
            # feats = list(range(day*4)) + list(range(day*17, day*18-1))
            # feats = list(range(day*(feat_num-1), day*feat_num-1))
            feats = [-1]
            pass

        if mode == 'test':
            # Testing data
            # data: 893 x 93 (40 states + day 1 (18) + day 2 (18) + day 3 (17))
            target = data[:, -1]
            data = data[:, feats]
            
            # print('data_test: ', data)
            # print('target_test: ', target)
            
            self.data = torch.FloatTensor(data)
            self.target = torch.FloatTensor(target)
        else:
            # Training data (train/dev sets)
            # data: 2700 x 94 (40 states + day 1 (18) + day 2 (18) + day 3 (18))
            target = data[:, -1]
            data = data[:, feats]
            # print('data_train: ', data)
            # print('target_train: ', target)
            
            # Splitting training data into train & dev sets
            # 0 → 2
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 1]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 1]
            
            # Convert data into PyTorch tensors
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # Normalize features (you may remove this part to see what will happen)
        # self.data[:, 40:] = \
        #     (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
        #     / self.data[:, 40:].std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]
        
        if print_process:
            print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
                  .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            # return self.data[index]
            return self.data[index], self.target[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)


# In[ ]:


def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = COVID19Dataset(path, mode=mode, target_only=target_only)  # Construct dataset
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)                            # Construct dataloader
    return dataloader


# In[ ]:


class NeuralNet(nn.Module):
    ''' A simple fully-connected deep neural network '''
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()

        # model 1: regression
        if model_num == 1:
            self.net = nn.Sequential(
                nn.Linear(input_dim, 1),
            )
        
        # model 2: One-layer perceptron
        elif model_num == 2:
            self.net = nn.Sequential(        
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            
        # model 3: multilayer perceptron(MLP)
        elif model_num == 3:
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            
        # model 4: deep neural network(DNN)
        elif model_num == 4:
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
        
        else:
            print("model selection error")
        

        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        ''' Calculate loss '''
        # TODO: you may implement L1/L2 regularization here
        return self.criterion(pred, target)


# In[ ]:


def train(tr_set, dv_set, model, config, device):
    ''' DNN training '''
    if print_process:
        print('\n')
    n_epochs = config['n_epochs']  # Maximum number of epochs

    # Setup optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hparas'])

    min_mse = 1000.0
    loss_record = {'train': [], 'dev': []}      # for recording training loss
    early_stop_cnt = 0
    final_epoch = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()                           # set model to training mode
        for x, y in tr_set:                     # iterate through the dataloader
            optimizer.zero_grad()               # set gradient to zero
            x, y = x.to(device), y.to(device)   # move data to device (cpu/cuda)
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
            mse_loss.backward()                 # compute gradient (backpropagation)
            optimizer.step()                    # update model with optimizer
            loss_record['train'].append(mse_loss.detach().cpu().item())

        # After each epoch, test your model on the validation (development) set.
        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            # Save model if your model improved
            min_mse = dev_mse
            if print_process:
                print('Saving model (epoch = {:4d}, loss = {:.4f})'.format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])  # Save model to specified path
            final_epoch = epoch + 1
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            # Stop training if your model stops improving for "config['early_stop']" epochs.
            break

    if print_process:
        print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record, final_epoch


# In[ ]:


def dev(dv_set, model, device):
    model.eval()                                # set model to evalutation mode
    total_loss = 0
    for x, y in dv_set:                         # iterate through the dataloader
        x, y = x.to(device), y.to(device)       # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
        total_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
    total_loss = total_loss / len(dv_set.dataset)              # compute averaged loss

    return total_loss


# In[ ]:


def test(tt_set, model, device):

    model.eval()                                # set model to evalutation mode
    testing_loss = 0
    for x, y in tt_set:                         # iterate through the dataloader
        x, y = x.to(device), y.to(device)       # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
        testing_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
    testing_loss = testing_loss / len(tt_set.dataset)            # compute averaged loss
    
    return testing_loss


# In[ ]:


def predict(set_type, model, device, preds=None, targets=None):
    
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in set_type:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()
    
    return preds, targets


# In[ ]:


def TRAINING():
    
    # Get the current available device ('cpu' or 'cuda')
    device = get_device()
    
    # Save the trained model to ./models/
    os.makedirs('models', exist_ok=True)  
    
    # Get model number
    globals()['model_num'] = config['model_num']
    
    # calculate feature number
    feat_num_list = pd.read_csv(f'./data/training/covid.train.1day.{read_name}.csv')
    globals()['feat_num'] = feat_num_list.shape[1]-1
    
    final_epochs = []
    train_final_loss_total = []
    testing_loss_total = []

    for i in range(len(day_num)):

        globals()['day'] = day_num[i]

        if print_process:
            print(f"---------------------------Training day {day} for mv = {mv}d---------------------------")

        '''Set random seed'''
        myseed = 42069  # set a random seed for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(myseed)
        torch.manual_seed(myseed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(myseed)

        '''Set data path'''
        tr_path = f'./data/training/covid.train.{day}day.{read_name}.csv'  # path to training data
        tt_path = f'./data/training/covid.test.{day}day.{read_name}.csv'   # path to testing data

        '''Load data and model'''
        tr_set = prep_dataloader(tr_path, 'train', config['batch_size'], target_only=target_only)
        dv_set = prep_dataloader(tr_path, 'dev', config['batch_size'], target_only=target_only)
        tt_set = prep_dataloader(tt_path, 'test', config['batch_size'], target_only=target_only)

        model = NeuralNet(tr_set.dataset.dim).to(device)  # Construct model and move to device

        '''Start Training'''
        model_loss, model_loss_record, final_epoch = train(tr_set, dv_set, model, config, device)

        '''Save plots'''
        plot_learning_curve(model_loss_record, title='deep model')

        '''Save model'''
        del model
        model = NeuralNet(tr_set.dataset.dim).to(device)
        os.makedirs('./models', exist_ok=True)                        # path to save models
        ckpt = torch.load(config['save_path'], map_location='cpu')  # Load your best model
        model.load_state_dict(ckpt)

        '''Record validation'''
        dv_preds, dv_targets = predict(dv_set, model, device)  # predict on the validation set
        plot_valid(dv_preds, dv_targets)                       # plot prediction on the validation set

        '''Record prediction'''
        testing_loss = test(tt_set, model, device)             # predict COVID-19 cases with your model
        tt_preds, tt_targets = predict(tt_set, model, device)  # predict on the testing set
        plot_pred(tt_preds, tt_targets)                        # plot prediction on the testing set

        globals()[f'preds_{day}'] = tt_preds
        globals()[f'targets_{day}'] = tt_targets

        '''Save results'''
        final_epochs.append(final_epoch)
        train_final_loss_total.append(model_loss)
        testing_loss_total.append(testing_loss)

        '''Print results'''
        if print_process:
            print('\nResult:')
            print(f'final_epoch_day{day} = {final_epoch}')
            print(f'train_final_loss_day{day} = {model_loss}')
            print(f'testing_loss_day{day} = {testing_loss}')
            print('\n')
        
    results_info = [final_epochs, train_final_loss_total, testing_loss_total]
    return results_info


# In[ ]:


def build_dataframe_of_results(results_info):
    
    final_epochs = results_info[0]
    train_final_loss_total = results_info[1]
    testing_loss_total = results_info[2]
    
    final_epoch_sr = pd.Series(final_epochs)
    train_final_loss_sr = pd.Series(train_final_loss_total)
    testing_loss_sr = pd.Series(testing_loss_total)

    result = pd.DataFrame({'exp_name': exp_name,
                           'day_num': day_num,
                           'target_only': target_only,
                           'model': config['model_num'],
                           'n_epochs': config['n_epochs'],
                           'batch_size': config['batch_size'],
                           'optimizer': config['optimizer'],
                           'lr': config['optim_hparas']['lr'],
                           'weight_decay': config['optim_hparas']['weight_decay'],
                           'betas': str(config['optim_hparas']['betas']),
                           'early_stop': config['early_stop'],
                           'final_epoch': final_epoch_sr, 
                           'train_final_loss': train_final_loss_sr, 
                           'testing_loss': testing_loss_sr})
    
    # save as excel
    excel_path = './results/data_experiment'
    os.makedirs(excel_path, exist_ok = True)
    result.to_excel(f'{excel_path}/data_exp_{exp_name}.xlsx', index = False)


# In[ ]:


def build_dataframe_of_real_data(target_range):
    
    # correct data
    correct_data_or_not = True
    
    if correct_data_or_not == True:
        
        for i in range(len(day_num)):

            preds = globals()[f'preds_{day_num[i]}']
            targets = globals()[f'targets_{day_num[i]}']

            preds, targets = correct_data_norm(preds, targets, target_range)
            # preds, targets = correct_data_scaled(preds, targets, target_range)

            globals()[f'preds_{day_num[i]}'] = preds
            globals()[f'targets_{day_num[i]}'] = targets

    # check preds
    for i in range(len(day_num)):
        preds = globals()[f'preds_{day_num[i]}']
        for j in range(len(preds)):
            if preds[j] < 0:
                preds[j] = 0
        globals()[f'preds_{day_num[i]}'] = preds
    
    # plot result
    for i in range(len(day_num)):
        globals()['day'] = day_num[i]
        preds = globals()[f'preds_{day_num[i]}']
        targets = globals()[f'targets_{day_num[i]}']
        plot_predicted_result(preds, targets)

    # bulid dataframe
    pred_output = pd.DataFrame()
    for i in range(len(day_num)):
        preds = globals()[f'preds_{day_num[i]}']
        targets = globals()[f'targets_{day_num[i]}']
        df_preds = pd.DataFrame({f'preds_{day_num[i]}': preds})
        df_targets = pd.DataFrame({f'targets_{day_num[i]}': targets})
        pred_output = pd.concat([pred_output, df_preds, df_targets], axis=1)

    
    # save as excel
    excel_path = './results/data_prediction'
    os.makedirs(excel_path, exist_ok = True)
    pred_output.to_excel(f'{excel_path}/data_pred_{exp_name}.csv', index = False)


# In[ ]:


def prediction_DNN(exp_num, day_num, mv_day, target_only, config, target_range, print_process):
    
    if print_process:
        print('\nStart training model...\n')
    
    for k in range(len(mv_day)):
        
        globals()['exp_name'] = f'exp_{exp_num}_num_smoothed_{mv_day[k]}d'
        globals()['read_name'] = f'num.smoothed.{mv_day[k]}d'
        globals()['day_num'] = day_num
        globals()['mv'] = mv_day[k]
        globals()['target_only'] = target_only
        globals()['config'] = config
        globals()['print_process'] = print_process
        
        results_info = TRAINING()
        build_dataframe_of_results(results_info)
        build_dataframe_of_real_data(target_range)
    
    print('Finished predicting.')

