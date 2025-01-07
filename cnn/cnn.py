import torch
import math, os
import tqdm
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

config = {
    'seed': 6666,
    'dataset_dir': "/home/liyuan/cpp/dp/data/cnn/food11/",
    'n_epochs': 10,
    'batch_size': 64,
    'learning_rate': 0.0003,
    'weight_decay': 1e-5,
    'early_stop': 300,
    'clip_flag': True,
    'save_path': './model.ckpt',
    'resnet_save_path': './resnet_model.ckpt'
}


class FoodDataset(torch.utils.data.Dataset):

  def __init__(self, path, tfm):
    super(FoodDataset).__init__()
    self.path = path
    self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
    print(f"One {path} sample", self.files[0])
    self.transform = tfm

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    fname = self.files[idx]
    im = Image.open(fname)
    im = self.transform(im)
    try:
      label = int(fname.split("/")[-1].split("_")[0])
    except:
      label = -1  # 测试集没有label
    return im, label


class Data:
  def __init__(self, train_data_dir, validate_data_dir, test_data_dir):
    self.train_data_dir = train_data_dir
    self.validate_data_dir = validate_data_dir
    self.test_data_dir = test_data_dir

  def load_pytorch_dataset(self, batch_size):
    test_tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_tfm = transforms.Compose([
        # 图片裁剪 (height = width = 128)
        transforms.Resize((128, 128)),
        # TODO:在这部分还可以增加一些图片处理的操作
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        # ToTensor() 放在所有处理的最后
        transforms.ToTensor(),
    ])
    train_set = FoodDataset(self.train_data_dir, tfm=train_tfm)
    self.train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=0, pin_memory=True)

    valid_set = FoodDataset(self.validate_data_dir, tfm=test_tfm)
    self.valid_loader = torch.utils.data.DataLoader(valid_set, batch_size, shuffle=True, num_workers=0, pin_memory=True)

    test_set = FoodDataset(self.test_data_dir, tfm=test_tfm)
    self.test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False, num_workers=0, pin_memory=True)

    test_set = FoodDataset(self.test_data_dir, tfm=train_tfm)
    self.test_loader_extra1 = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False, num_workers=0, pin_memory=True)

    test_set = FoodDataset(self.test_data_dir, tfm=train_tfm)
    self.test_loader_extra2 = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False, num_workers=0, pin_memory=True)

    test_set = FoodDataset(self.test_data_dir, tfm=train_tfm)
    self.test_loader_extra3 = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False, num_workers=0, pin_memory=True)


class ImageClassifier(torch.nn.Module):
  def __init__(self):
    super(ImageClassifier, self).__init__()
    # input 維度 [3, 128, 128]
    self.cnn = nn.Sequential(
        nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

        nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

        nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

        nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]

        nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
    )
    self.fc = nn.Sequential(
        nn.Linear(512*4*4, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 11)
    )

  def forward(self, x):
    out = self.cnn(x)
    out = out.view(out.size()[0], -1)
    return self.fc(out)


class Train:
  def __init__(self, data: Data, model: ImageClassifier, device, ecpochs, modelpath):
    self.data = data
    self.model = model
    self.device = device
    self.epochs = ecpochs
    self.modelpath = modelpath
    self.best_loss = math.inf
    self.tqdmobj = None
    self.tqdmpost = {}

  def do_train(self):
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # 交叉熵计算时，label范围为[0, n_classes-1]
    optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    device = self.device

    self.tqdmobj = tqdm.tqdm(range(self.epochs))
    for epoch in self.tqdmobj:
      self.tqdmobj.set_description('epoch[%s/%s]' % (epoch, self.epochs))
      self.tqdmobj.set_postfix(self.tqdmpost)
      self.model.train()
      loss_record = []
      train_accs = []
      for x, y in self.data.train_loader:
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        pred = self.model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        acc = (pred.argmax(dim=-1) == y.to(device)).float().mean()
        l_ = loss.detach().item()
        loss_record.append(l_)
        train_accs.append(acc.detach().item())
      mean_train_acc = sum(train_accs) / len(train_accs)
      mean_train_loss = sum(loss_record)/len(loss_record)
      self.tqdmpost['TrainLoss'] = mean_train_loss
      self.tqdmpost['TrainAcc'] = mean_train_acc
      self.do_validate()

  def do_validate(self):
    self.model.eval()  # 设置模型为评估模式
    device = self.device
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # 交叉熵计算时，label范围为[0, n_classes-1]
    loss_record = []
    test_accs = []
    for x, y in self.data.valid_loader:
      x, y = x.to(device), y.to(device)
      with torch.no_grad():
        pred = self.model(x)
        loss = criterion(pred, y)
        acc = (pred.argmax(dim=-1) == y.to(device)).float().mean()
      loss_record.append(loss.item())
      test_accs.append(acc.detach().item())
    mean_valid_acc = sum(test_accs) / len(test_accs)
    mean_valid_loss = sum(loss_record)/len(loss_record)
    self.tqdmpost['ValidLoss'] = mean_valid_loss
    self.tqdmpost['ValidAcc'] = mean_valid_acc
    if mean_valid_loss < self.best_loss:
      self.best_loss = mean_valid_loss
      torch.save(model.state_dict(), self.modelpath)  # 保存最优模型
      print('Saving model with loss {:.3f}...'.format(self.best_loss))


if __name__ == '__main__':
  device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
  import common
  common.same_seed()

  current_directory_path = os.path.dirname(os.path.abspath(__file__))
  datadir = current_directory_path + '/../data/cnn/food11'
  data = Data(datadir+'/training/', datadir+'/validation/', datadir+'/test/')
  data.load_pytorch_dataset(config['batch_size'])

  model = ImageClassifier().to(device)

  common.model_plot(ImageClassifier, torch.randn(1, 3, 128, 128).requires_grad_(True))

  train = Train(data, model, device, config['n_epochs'], config['save_path'])
  train.do_validate()
  train.do_train()

