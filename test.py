# from src.vit import *
# m =  vit_2_4_32()
# print(m)
# from torchsummary import summary
#
# summary(m.cuda(),(3,32,32))

import torch
from torch_batch_svd import svd

A = torch.rand(100, 3, 3).cuda()
u, s, v = svd(A)
u, s, v = torch.svd(A)