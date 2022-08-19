import torch
from mri_pituitary.utils import extract_data, insert_data, none_data
from mri_pituitary.metrics import compute_metrics


class ObjectiveFunction:

    def __init__(self, net, loss, alpha=1e-3):
        self.net = net
        self.loss = loss
        self.loss.reduction = 'sum'
        self.alpha = alpha

        self.info = dict()
        # self.info['header'] = ('loss', 'acc', 'red', 'green', 'blue', 'back', 'avg.')
        # self.info['frmt'] = '{:<15.4e}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}'

        self.info['header'] = ()
        self.info['frmt'] = ''

    def evaluate(self, p, x, y, do_gradient=False):
        (Jc, dJc) = (None, None)

        # averaging factor
        beta = 1 / x.shape[0]

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

            Jc = beta * misfit.detach() + reg
            dJc = beta * g + dreg
        else:
            self.net.eval()
            with torch.no_grad():
                out = self.net(x)
                misfit = self.loss(out, y)

                reg = 0.5 * self.alpha * torch.norm(p) ** 2

                Jc = beta * misfit + reg

        return Jc, dJc

    def get_metrics(self, p, images_train, masks_train, images_val=None, masks_val=None):
        self.net.eval()

        insert_data(self.net, p)
        with torch.no_grad():
            Jc_train, acc_train, dice_train = compute_metrics(images_train, masks_train, self.net, self.loss)
            values = [Jc_train, acc_train] + dice_train + [sum(dice_train) / len(dice_train)]
            values = values[:len(self.info['header'])]

            if images_val is not None and masks_val is not None:
                Jc_val, acc_val, dice_val = compute_metrics(images_val, masks_val, self.net, self.loss)
                values += [Jc_val, acc_val] + dice_val + [sum(dice_val) / len(dice_val)]

        return values


if __name__ == '__main__':
    import torch.nn as nn
    from mri_pituitary.utils import convert_to_base
    # do gradient check

    # create data
    N, m, n = 100, 3, 4
    x = torch.randn(N, m)

    y = torch.randn(N, n)
    net = nn.Sequential(nn.Linear(m, 10), nn.ReLU(), nn.Linear(10, n))
    loss = nn.MSELoss()
    #
    # y = torch.randint(3, (N,))
    # net = nn.Sequential(nn.Linear(m, 10), nn.ReLU(), nn.Linear(10, 3))
    # loss = nn.CrossEntropyLoss()
    #
    #
    # N, C, m, n = 100, 2, 16, 16
    # x = torch.randn(N, C, m, n)
    #
    # y = torch.randn(N, 3, 10, 10)
    # net = nn.Sequential(nn.Conv2d(C, 5, (3, 3)), nn.ReLU(), nn.Conv2d(5, 3, (5, 5)))
    # loss = nn.CrossEntropyLoss(weight=torch.tensor((0, 1, 1e-2)))

    # create network
    alpha = 1e0
    f = ObjectiveFunction(net, loss, alpha=alpha)

    p = extract_data(net, 'data')
    f0, df0 = f.evaluate(p, x, y, do_gradient=True)

    d = torch.randn_like(p)
    dfd = torch.dot(df0.view(-1), d.view(-1))

    N = 15

    headers = ('h', 'E0', 'E1')
    print(('{:<20s}' * len(headers)).format(*headers))

    err0 = []
    err1 = []
    for k in range(N):
        h = 2 ** (-k)
        f1 = f.evaluate(p + h * d, x, y, do_gradient=False)[0]

        err0.append(torch.norm(f0 - f1))
        err1.append(torch.norm(f0 + h * dfd - f1))

        printouts = convert_to_base((err0[-1], err1[-1]))
        print(((1 + len(printouts) // 2) * '%0.2f x 2^(%0.2d)\t\t') % ((1, -k) + printouts))




