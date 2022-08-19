import torch
import torch.nn as nn
from mri_pituitary.objective_function import ObjectiveFunction
from mri_pituitary.utils import extract_data, insert_data, none_data, get_num_parameters, seed_everything


torch.set_default_dtype(torch.float64)

seed_everything(42)

# # create data
# N, m, n = 100, 3, 4
# x = torch.randn(N, m)
# y = torch.randn(N, n)
#
# # create network
# net = nn.Linear(m, n)

# create loss
# loss = nn.MSELoss()



N, C, m, n = 100, 2, 16, 16
x = torch.randn(N, C, m, n)

y = torch.rand(N, 2, 10, 10)
y = y / y.sum(dim=1, keepdim=True)
y = 1 * (y >= 0.5)
y = torch.cat((y, 1 - y.sum(dim=1, keepdim=True)), dim=1).to(torch.float32)
net = nn.Sequential(nn.Conv2d(C, 5, (3, 3)), nn.Tanh(), nn.Conv2d(5, 3, (5, 5)))
loss = nn.CrossEntropyLoss(weight=torch.tensor((1, 1, 1)))

# create objective function
alpha = 0
f = ObjectiveFunction(net, loss, alpha=alpha)

# check evaluation
out = net(x)
val = loss(out, y)
print(val)

p = extract_data(net, 'data')
val2 = f.evaluate(p, x, y, do_gradient=False)[0]
print(val2)
print(torch.norm(val - val2))


# check gradient
# check evaluation
none_data(net, 'grad')
out = net(x)
val = loss(out, y)
val.backward()

g = extract_data(net, 'grad')
print(g)

none_data(net, 'grad')
val2, g2 = f.evaluate(p, x, y, do_gradient=True)
print(g2)

print(torch.norm(g - g2))




#%%

import math
from mri_pituitary.lbfgs import LBFGS


# A = torch.cat((x, torch.ones(x.shape[0], 1)), dim=1)
# I = torch.eye(m + 1)
# z = torch.zeros(m + 1, n)
# z = torch.linalg.lstsq(torch.cat((A, math.sqrt(alpha) * I), dim=0), torch.cat((y, z), dim=0))
# x_opt = z.solution
#
# z = torch.linalg.solve(A.T @ A + alpha * I, A.T @ y)

# tmp = torch.cat((x_opt[:-1].T.reshape(-1), x_opt[-1].reshape(-1)))
# f_opt = f.evaluate(tmp, x, y)

none_data(net, 'grad')
n_params = get_num_parameters(net)
opt = LBFGS(n_params, m=10)

tmp, _ = opt.solve(f, p, x, y)

# # weight
# print(torch.norm(tmp[:-n].reshape(-1) - x_opt[:-1].T.reshape(-1)))
#
# # bias
# print(torch.norm(tmp[-n:].reshape(-1) - x_opt[-1].reshape(-1)))