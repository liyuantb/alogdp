import os, random 
import torch
import numpy as np

def same_seed(seed=0): 
  np.random.seed(seed)
  random.seed(seed)
  # CPU
  torch.manual_seed(seed) 
  # GPU
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed)
      torch.cuda.manual_seed(seed) 
  # python 全局
  os.environ['PYTHONHASHSEED'] = str(seed) 
  # cudnn
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.enabled = False

