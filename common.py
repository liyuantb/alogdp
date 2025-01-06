import os, random 
import torch
import numpy as np
from torchviz import make_dot

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

def model_plot(model_class, input_sample):
  if isinstance(model_class, torch.nn.Module):
   clf = model_class 
  else:
   clf = model_class()
  y = clf(input_sample) 
  clf_view = make_dot(y, params=dict(list(clf.named_parameters()) + [('x', input_sample)]))
  clf_view.render("model_graph")
  return clf_view

