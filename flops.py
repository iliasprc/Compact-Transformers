
import numpy as np
import torch
import torch.nn as nn
from src import *
from src.grassmanian_vit import *
from src.img_gm_vit import *
from src.manifold_vit import *



m  = manifold_cct_7_3x2_32(attention_type='all')


from pthflops import count_ops

inp = torch.rand(1,8,64,64)
inp = torch.rand(1,3,32,32)
print(m)
model_parameters = filter(lambda p: p.requires_grad, m.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
# Count the number of FLOPs
count_ops(m, inp)


print(f' Number of params :  {params}')
 
