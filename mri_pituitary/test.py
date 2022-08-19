import torch
import torch.nn as nn
from mri_pituitary.objective_function import ObjectiveFunction
from mri_pituitary.utils import extract_data, insert_data, none_data, get_num_parameters, seed_everything


torch.set_default_dtype(torch.float64)

seed_everything(42)

# # create data
N, m, n = 100, 6, 4
x = torch.randn(N, m)
y = torch.randn(N, n)

# create network
net = nn.Linear(m, n)

# create loss
loss = nn.MSELoss(reduction='sum')



# N, C, m, n = 100, 2, 16, 16
# x = torch.randn(N, C, m, n)
#
# y = torch.rand(N, 2, 10, 10)
# y = y / y.sum(dim=1, keepdim=True)
# y = 1 * (y >= 0.5)
# y = torch.cat((y, 1 - y.sum(dim=1, keepdim=True)), dim=1).to(torch.float32)
# net = nn.Sequential(nn.Conv2d(C, 5, (3, 3)), nn.Tanh(), nn.Conv2d(5, 3, (5, 5)))
# loss = nn.CrossEntropyLoss(weight=torch.tensor((1, 1, 1)))

# create objective function
alpha = 1e-2
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


A = torch.cat((x, torch.ones(x.shape[0], 1)), dim=1)
I = torch.eye(m + 1)
nn = A.shape[0]

z = torch.zeros(m + 1, n)
z = torch.linalg.lstsq(torch.cat((1 / math.sqrt(nn) * A, math.sqrt(alpha) * I), dim=0),
                       torch.cat((1 / math.sqrt(nn) * y, z), dim=0))
x_opt = z.solution

z = torch.linalg.solve((1 / nn) * A.T @ A + alpha * I, (1 / nn) * A.T @ y)

torch_sol = torch.cat((x_opt[:-1].T.reshape(-1), x_opt[-1].reshape(-1)))
f_opt = f.evaluate(torch_sol, x, y)

none_data(net, 'grad')
n_params = get_num_parameters(net)
opt = LBFGS(n_params, m=1000)

my_sol, _ = opt.solve(f, p, x, y)

print(torch.norm(my_sol.view(-1) - torch_sol.view(-1)))

# # weight
print(torch.norm(my_sol[:-n].reshape(-1) - x_opt[:-1].T.reshape(-1)))

# bias
print(torch.norm(my_sol[-n:].reshape(-1) - x_opt[-1].reshape(-1)))
