import os, sys, math
import torch
import pandas as pd
import numpy as np

config = {
  'seed': 5201314,      # 随机种子，可以自己填写. :)
  'select_all': True,   # 是否选择全部的特征
  'valid_ratio': 0.2,   # 验证集大小(validation_size) = 训练集大小(train_size) * 验证数据占比(valid_ratio)
  'n_epochs': 1000,     # 数据遍历训练次数           
  'batch_size': 256, 
  'learning_rate': 1e-5,              
  'early_stop': 400,    # 如果early_stop轮损失没有下降就停止训练.     
  'save_path': './model.ckpt'  # 模型存储的位置
}


class COVID19Dataset(torch.utils.data.Dataset):
  def __init__(self, x, y=None):
    if y is None:
      self.y = y
    else:
      self.y = torch.FloatTensor(y)
    self.x = torch.FloatTensor(x)

  def __getitem__(self, idx):
    if self.y is None:
      return self.x[idx]
    return self.x[idx], self.y[idx]

  def __len__(self):
    return len(self.x)


class Data:
  def __init__(self, train_data_file, test_data_file):
    self.train_data_file = train_data_file
    self.test_data_file = test_data_file

  def train_valid_split(self, train_data, valid_ratio):
      valid_set_size = int(valid_ratio * len(train_data)) 
      train_set_size = len(train_data) - valid_set_size
      train_set, valid_set = torch.utils.data.random_split(train_data, [train_set_size, valid_set_size])
      return np.array(train_set), np.array(valid_set)

  def load_data(self, valid_ratio):
    train_df, test_df = pd.read_csv(self.train_data_file), pd.read_csv(self.test_data_file)
    train_data = train_df.values 
    self.train_data, self.valid_data = self.train_valid_split(train_data, valid_ratio)
    self.test_data = test_df.values

  def select_feat_label(self, select_all=True):
    # 获取最后一列标记(y)
    y_train, y_valid = self.train_data[:, -1], self.valid_data[:, -1]
    # 去掉最后一列标记(x)
    raw_x_train, raw_x_valid, raw_x_test = self.train_data[:, :-1], self.valid_data[:, :-1], self.test_data
    if select_all:
      feat_idx = list(range(raw_x_train.shape[1]))
    else:
      feat_idx = [0, 1, 2, 3, 4,]  # TODO: 选择需要的特征 ，这部分可以自己调研一些特征选择的方法并完善.
    self.x_train_data = raw_x_train[:,feat_idx]
    self.y_train_data = y_train
    self.x_valid_data = raw_x_valid[:,feat_idx]
    self.y_valid_data = y_valid
    self.x_test_data = raw_x_test[:,feat_idx]

  def load_pytorch_dataset(self, batchsize): 
    train_dataset = COVID19Dataset(self.x_train_data, self.y_train_data) 
    valid_dataset = COVID19Dataset(self.x_valid_data, self.y_valid_data)
    test_dataset = COVID19Dataset(self.x_test_data, None)
    # 使用Pytorch中Dataloader类按照Batch将数据集加载
    self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True, pin_memory=True)
    self.valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batchsize, shuffle=True, pin_memory=True)
    self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=False, pin_memory=True)


class CovidModel(torch.nn.Module):
  def __init__(self, n_feature, hiddens):
    super(CovidModel, self).__init__()
    self.layers = torch.nn.Sequential()
    last = None
    for x in hiddens:
      if last == None:
        module = torch.nn.Linear(n_feature, x)
      else:
        module = torch.nn.Linear(last, x)
      last = x
      self.layers.append(module)
      self.layers.append(torch.nn.ReLU())
    module = torch.nn.Linear(last, 1)
    self.layers.append(module)

  def forward(self, x):
    print(x.size())
    x = self.layers(x)
    x = x.squeeze(1) # (B, 1) -> (B)
    return x


class Train:
  def __init__(self, data:Data, model:CovidModel, device, batchsize, ecpochs, modelpath):
    self.data = data
    self.model = model
    self.device = device
    self.batchsize = batchsize
    self.epochs = ecpochs
    self.modelpath = modelpath
    self.best_loss = math.inf

  def do_train(self):
    criterion = torch.nn.MSELoss(reduction='mean') # 损失函数的定义
    optimizer = torch.optim.SGD(self.model.parameters(), lr=config['learning_rate']) 
    device = self.device
    for epoch in range(self.epochs):
      self.model.train() # 训练模式
      train_loss_record = []

      for x, y in self.data.train_loader:
        x, y = x.to(device), y.to(device)   # 将数据一到相应的存储位置(CPU/GPU)
        pred = self.model(x)             
        loss = criterion(pred, y)
        optimizer.zero_grad()               # 将梯度置0.
        loss.backward()                     # 反向传播 计算梯度.
        optimizer.step()                    # 更新网络参数
        train_loss_record.append(loss.detach().item())
      mean_train_loss = sum(train_loss_record)/len(train_loss_record)
      print(f"in epoch[{epoch}]/[{self.epochs}] trainloss[{mean_train_loss }]")
      self.do_validate()


  def do_validate(self):
    criterion = torch.nn.MSELoss(reduction='mean') # 损失函数的定义
    device = self.device
    self.model.eval() # 将模型设置成 evaluation 模式.
    valid_loss_record = []
    for x, y in self.data.valid_loader:
      x, y = x.to(device), y.to(device)
      with torch.no_grad():
        pred = self.model(x)
        loss = criterion(pred, y)
      valid_loss_record.append(loss.item())
    mean_valid_loss = sum(valid_loss_record)/len(valid_loss_record)
    print(f"validloss[{mean_valid_loss}]")
    if mean_valid_loss < self.best_loss:
      self.best_loss = mean_valid_loss
      torch.save(self.model.state_dict(), self.modelpath) # 模型保存
      print('Saving model with loss {:.3f}...'.format(self.best_loss))



if __name__ == '__main__':
  device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
  print(device)
  import common 
  common.same_seed()

  current_directory_path = os.path.dirname(os.path.abspath(__file__))
  datadir = current_directory_path + '/../data/dnn'
  data = Data(datadir+'/covid.train.csv', datadir+'/covid.test.csv')
  data.load_data(config['valid_ratio'])
  data.select_feat_label(select_all=True)
  data.load_pytorch_dataset(config['batch_size'])

  n_feaures = len(data.x_train_data[0])
  print(f"n_feaures[{n_feaures}]")
  hiddens = [n_feaures, 10, 9]
  model = CovidModel(n_feaures, hiddens).to(device) # 将模型和训练数据放在相同的存储位置(CPU/GPU)
  common.model_plot(model, torch.randn(1, n_feaures).requires_grad_(True).to(device))

  train = Train(data, model, device, config['batch_size'], config['n_epochs'], config['save_path'])
  train.do_validate()
  train.do_train()