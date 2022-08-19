import torch
from mri_pituitary.utils import extract_data, insert_data, none_data
from mri_pituitary.train import compute_metrics


class ObjectiveFunction:

    def __init__(self, net, loss, alpha=1e-3):
        self.net = net
        self.loss = loss
        self.alpha = alpha

    def evaluate(self, p, x, y, do_gradient=False):
        (Jc, dJc) = (None, None)

        # insert parameters
        none_data(self.net, 'grad')
        insert_data(self.net, p)
        if do_gradient:
            self.net.train()
            out = self.net(x)
            misfit = self.loss(out, y)
            misfit.backward()

            g = extract_data(self.net, 'grad')
            reg = 0.5 * self.alpha * torch.norm(p) ** 2
            dreg = self.alpha * p

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

    def get_metrics(self, p, images_train, masks_train, images_val=None, masks_val=None):
        self.net.eval()

        insert_data(self.net, p)
        with torch.no_grad():
            Jc_train, acc_train, dice_train = compute_metrics(images_train, masks_train, self.net, self.loss)
            values = [Jc_train, acc_train,
                      dice_train[0].item(), dice_train[1].item(), dice_train[2].item(), dice_train[3].item(),
                      dice_train.mean().item()]

            if images_val is not None and masks_val is not None:
                Jc_val, acc_val, dice_val = compute_metrics(images_val, masks_val, self.net, self.loss)
                values += [Jc_val, acc_val, dice_val[0].item(), dice_val[1].item(), dice_val[2].item(), dice_val[3].item(),
                           dice_val.mean().item()]

        return values

