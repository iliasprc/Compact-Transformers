# from src.vit import *
# m =  vit_2_4_32()
# print(m)
# from torchsummary import summary
#
# summary(m.cuda(),(3,32,32))

#
# A = torch.rand(100, 3, 3).cuda()
# u, s, v = svd(A)
# u, s, v = torch.svd(A)
import numpy as np

from src import *
from src.grassmanian_vit import *
from src.img_gm_vit import *
from src.manifold_vit import *

m = grassmanian_vit_6_4_32()
m = manifold_vit_6_4_32()
m = img_riem_vit_6_4_32()
m = img_gm_vit_6_4_32()
m = manifold_cct_7_3x1_32()
# m = manifold_vit_nano_12_p16()
# m = manifold_vit_small_12_p16()

# m = vit_6_4_32()

from pthflops import count_ops

inp = torch.rand(1, 3, 32, 32)
# inp = torch.rand(1,3,224,224)

model_parameters = filter(lambda p: p.requires_grad, m.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
# Count the number of FLOPs
count_ops(m, inp)
print(f' Number of params :  {params}')
print(m)
m = torch.nn.MaxPool2d(3, 2, 1)
o = m(m(inp))
print(o.shape)
