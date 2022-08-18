import torch
from mri_pituitary.utils import extract_data, insert_data, none_grad


class ObjectiveFunction:

    def __init__(self, net, loss, alpha=1e-3):
        self.net = net
        self.loss = loss
        self.alpha = alpha
        self.info = None

    def evaluate(self, p, x, y, do_gradient=False):
        self.info = None
        (Jc, dJc) = (None, None)

        # insert parameters
        insert_data(self.net, p)
        if do_gradient:
            self.net.train()
            none_grad(self.net)
            out = self.net(x)
            misfit = self.loss(out, y)
            misfit.backward()

            g = extract_data(self.net, 'grad')
            reg = 0.5 * self.alpha * torch.norm(p) ** 2
            dreg = self.alpha * g

            Jc = misfit + reg
            dJc = g + dreg
        else:
            self.net.eval()
            with torch.no_grad():
                out = self.net(x)
                misfit = self.loss(out, y)

                reg = 0.5 * self.alpha * torch.norm(p) ** 2
                Jc = misfit + reg

        return Jc, dJc


