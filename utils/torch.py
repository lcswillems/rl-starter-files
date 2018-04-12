import torch
import numpy as np
from torch.autograd import Variable

use_gpu = torch.cuda.is_available()
DoubleTensor = torch.DoubleTensor
LongTensor = torch.LongTensor