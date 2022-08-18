import torch
import math

# https://github.com/minrk/scipy-1/blob/master/scipy/optimize/linesearch.py

class WolfeLineSearch:

    def __init__(self):
        self.max_iter = 50
        self.c1 = 1e-4
        self.c2 = 0.9
        self.alpha_max = 5
        self.max_zoom_iter = 50

    def search(self, obj_fctn, p, d, x, y, alpha=1.0):
        phi0, dphi0 = self.phi(obj_fctn, p, d, x, y, 0, do_gradient=True)
        alpha_old, phi_old, dphi_old = 0, phi0.clone(), dphi0.clone()

        iter = 0
        while iter < self.max_iter:
            phic, dphic = self.phi(obj_fctn, p, d, x, y, alpha, do_gradient=True)

            if (phic > phi0 + self.c1 * alpha * dphi0) or (phic > phi_old and iter > 0):
                print('here1')
                alpha_opt = self.zoom(obj_fctn, p, d, x, y, phi0, dphi0, alpha_old, phi_old, dphi_old, alpha, phic, dphic)
                if isinstance(alpha, torch.Tensor):
                    alpha_opt = alpha_opt.item()

                return alpha_opt

            if abs(dphic) <= -self.c2 * dphi0:
                print('here2')
                if isinstance(alpha, torch.Tensor):
                    alpha_opt = alpha.item()
                else:
                    alpha_opt = alpha
                return alpha_opt

            if dphic >= 0:
                print('here3')
                alpha_opt = self.zoom(obj_fctn, p, d, x, y, phi0, dphi0, alpha, phic, dphic, alpha_old, phi_old, dphi_old)
                if isinstance(alpha_opt, torch.Tensor):
                    alpha_opt = alpha_opt.item()
                return alpha_opt

            alpha_old, phi_old, dphi_old = deepcopy(alpha), phic.clone(), dphic.clone()
            alpha = (alpha + self.alpha_max) / 2

            iter += 1

        # did not converge
        return 0

    def zoom(self, obj_fctn, p, d, x, y, phi0, dphi0, alphaLo, phiLo, dphiLo, alphaHi, phiHi, dphiHi):
        iter = 0
        while iter < self.max_zoom_iter:
            alpha = self.interpolate(alphaLo, phiLo, dphiLo, alphaHi, phiHi, dphiHi)
            # alpha = 0.5 * (alphaLo + alphaHi)

            phic, dphic = phi(obj_fctn, p, d, x, y, alpha, do_gradient=True)
            if phic > phi0 + self.c1 * alpha or phic >= phiLo:
                if isinstance(alpha, torch.Tensor):
                    alpha = alpha.item()

                alphaHi = alpha
            else:
                if abs(dphic) <= -self.c2 * dphi0:
                    if isinstance(alpha, torch.Tensor):
                        alpha = alpha.item()
                    return alpha

                if dphic * (alphaHi - alphaLo) >= 0:
                    alphaHi = alphaLo

                alphaLo = alpha
                phiLo = phic

            iter += 1

        # did not converge
        return 0


def phi(obj_fctn, p, d, x, y, alpha, do_gradient=False):
    phic, dphic = None, None
    if do_gradient:
        phic, dfc = obj_fctn.evaluate(p + alpha * d, x, y, do_gradient=True)
        dphic = torch.dot(dfc.view(-1), d.view(-1))
    else:
        phic, _ = obj_fctn.evaluate(p + alpha * d, x, y, do_gradient=False)

    return phic, dphic


def interpolate(alphaLo, phiLo, dphiLo, alphaHi, phiHi, dphiHi):
    alpha = -(alphaLo ** 2 * dphiHi) / (2 * (phiLo - phiHi - alphaLo * dphiHi))

    myEps = 0.1 * abs(alphaHi - alphaLo)
    if (alpha <= myEps + min(alphaHi, alphaLo)) or (alpha >= max(alphaHi, alphaLo) - myEps):
        alpha = 0.5 * (alphaHi + alphaLo)

    return alpha


def cubicmin(x, fx, dfx, y, fy, z, fz):
    """
    Finds the minimizer for a cubic polynomial that goes through the
    points (x,fx), (y,fy), and (z,fz) with derivative at x of dfx.
    If no minimizer can be found return None

    f(a) = A * x^3 + B*x^2 + C*x + D
    """

    # find linear and constant coefficients
    c = dfx
    d = fx

    # compute differences for easy calculations
    yx = y - x
    zx = z - x
    yz = y - z

    # differences between function evaluated at y and z
    rhs1 = fy - d - c * yx
    rhs2 = fz - d - c * zx

    # compute leading coefficient
    a = (rhs1 / yx ** 2 - rhs2 / zx ** 2) / yz

    # compute
    b = rhs1 / yx ** 2 - a * yx

    # adjust coefficients
    d += -c * x + b * x ** 2 - a * x ** 3
    c += -2 * b * x + 3 * a * x ** 2
    b += -3 * a * x

    # find minimizer
    D = 4 * b ** 2 - 12 * a * c
    if D > 0:
        xmin1 = (-2 * b - math.sqrt(D)) / (6 * a)
        xmin2 = (-2 * b + math.sqrt(D)) / (6 * a)

        # check sign (or can use second derivative)
        if 6 * a * xmin1 + 2 * b > 0:
            xmin = xmin1
        else:
            xmin = xmin2

        fmin = a * xmin ** 3 + b * xmin ** 2 + c * xmin + d

        # adjust minimizer
        # xmin -= x
    else:
        xmin = None
        fmin = None

    return xmin, fmin, (a, b, c, d)


def quadraticmin(x, fx, dfx, y, fy):
    c = fx
    b = dfx

    yx = y - x
    a = (fy - b * yx - c) / yx ** 2

    if a > 0:

        # adjust coefficients
        c += -b * x + a * x ** 2
        b += -2 * a * x

        # find minimizer
        xmin = -b / 2 * a
        fmin = a * xmin ** 2 + b * xmin + c
    else:
        xmin = None
        fmin = None

    return xmin, fmin, (a, b, c)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch

    # # easy case
    # a, b, c, d = -1.0, -2.0, 3.0, 4.0
    #
    # t = torch.linspace(-3, 3, 50)
    # ft = a * t ** 3 + b * t ** 2 + c * t + d
    # dft = 3 * a * t ** 2 + 2 * b * t + c
    #
    # idx = torch.randperm(t.numel())[:3]
    # x = t[idx]
    # fx = ft[idx]
    # dfx = dft[idx]
    #
    # xmin, fmin, coeff = cubicmin(x[0], fx[0], dfx[0], x[1], fx[1], x[2], fx[2])
    #
    # print(coeff)
    # plt.figure()
    # plt.plot(t, ft, linewidth=2)
    # plt.plot(t, dft, linewidth=2)
    # plt.plot(xmin, fmin, 'o')
    # plt.show()


    # easy case
    a, b, c = 1.0, -2.0, 3.0

    t = torch.linspace(-3, 3, 50)
    ft = a * t ** 2 + b * t + c
    dft = 2 * a * t + b

    idx = torch.randperm(t.numel())[:2]
    x = t[idx]
    fx = ft[idx]
    dfx = dft[idx]

    xmin, fmin, coeff = quadraticmin(x[0], fx[0], dfx[0], x[1], fx[1])

    print(coeff)
    plt.figure()
    plt.plot(t, ft, linewidth=2)
    plt.plot(t, dft, linewidth=2)
    plt.plot(xmin, fmin, 'o')
    plt.show()





