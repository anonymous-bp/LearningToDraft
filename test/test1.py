import numpy as np
import sys
sys.path.append('..')
sys.path.append('../src')
from config import config
from src.overall_predictor import overall_network, overall_predictor
import torch
batch_size = 1
ban_single_camp = 4
pick_single_camp = 5
rounds = 1 
xx = overall_predictor(config)
xx.load_model('models.copy', 'checkpoint')
x = xx.network
y = torch.ones([1, 11], dtype=torch.long).cuda()
y[0][0] = 11
y[0][1] = 0
y[0][2] = 22
y[0][3] = 98
y[0][4] = 98

y[0][5] = 2
y[0][6] = 56
y[0][7] = 57
y[0][8] = 5
y[0][9] = 98

y[0][10] = 0
print(y)
print(x)
print(y.shape)
a, b = x(y)
print(a)
print(b)
print(np.argmax(b.cpu().detach().numpy()))

