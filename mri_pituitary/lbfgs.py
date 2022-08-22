
import torch
from copy import deepcopy
import math


class LBFGS:

    def __init__(self, n, m=10,
                 max_iter=100,
                 gamma=1.0,
                 alpha=1.0,
                 atol=1e-10,
                 rtol=1e-10,
                 dtype=None, device=None):
        factor_kwargs = {'dtype': dtype, 'device': device}
        self.n = n      # number of parameters
        self.m = m      # memory depth
        self.S = torch.zeros(m, n, **factor_kwargs)     # storage for difference in states
        self.Y = torch.zeros(m, n, **factor_kwargs)    # storage for difference in gradients
        self.rho = torch.zeros(m, 1, **factor_kwargs)       # curavture info
        self.k = 0
        self.gamma = alpha
        self.alpha = gamma
        self.atol = atol
        self.rtol = rtol
        self.max_iter = max_iter
        self.ls = WolfeLineSearch()
        # self.ls = ArmijoLineSearch()
        self.comp_metrics = True

    def solve(self, obj_fctn, p, x, y, x_val=None, y_val=None):

        info = dict()
        info['header'] = ('iter', 'f', '|df|', '|df|/|df0|', '|x1-x0|', 'alpha', 'ls_iter', 'zoom_iter')
        info['header'] += obj_fctn.info['header']

        info['frmt'] = '{:<15d}{:<15.4e}{:<15.4e}{:<15.4e}{:<15.4e}{:<15.4e}{:<15d}{:<15d}'
        info['frmt'] += obj_fctn.info['frmt']

        if x_val is not None and y_val is not None:
            info['header'] += obj_fctn.info['header']
            info['frmt'] += obj_fctn.info['frmt']

        info['values'] = torch.empty(0, len(info['header']))

        print((len(info['header']) * '{:<15s}').format(*info['header']))

        # ============
        # initial evaluation
        f0, df0 = obj_fctn.evaluate(p, x, y, do_gradient=True)
        nrmdf0 = torch.norm(df0).item()
        p_old, f, df = p.clone(), f0.clone(), df0.clone()
        f_old, df_old = f.clone(), df.clone()
        alpha = 0.0

        dfnrm = torch.norm(df0).item()
        values = [self.k, f0.item(), dfnrm, dfnrm / nrmdf0, torch.norm(p - p_old).item(), alpha, 0, 0]
        values += obj_fctn.get_metrics(p, x, y, x_val, y_val)
        print(info['frmt'].format(*values))

        # ============
        # main iteration
        while dfnrm > self.atol and dfnrm / nrmdf0 > self.rtol and self.k < self.max_iter:

            if self.k > 0:
                # get curvature info
                self.gamma = torch.dot(self.S[-1], self.Y[-1]) / torch.dot(self.Y[-1], self.Y[-1])

            # compute step
            d = self.two_loop_recursion(-df)

            # check for descent direction
            if torch.dot(d.view(-1), df.view(-1)) > 0:
                print('here')
                d = -df

            # perform line search
            alpha, ls_iter, zoom_iter = self.ls.search(obj_fctn, p, d, x, y)
            # alpha, k = self.ls.search(obj_fctn, p, d, f, df, x, y, alpha)

            # update parameters
            p += alpha * d

            # evaluate
            f, df = obj_fctn.evaluate(p, x, y, do_gradient=True)

            # update storage (k - m to k)
            self.S = torch.cat((self.S[1:], alpha * d.view(1, -1)), dim=0)
            self.Y = torch.cat((self.Y[1:], (df - df_old).view(1, -1)), dim=0)
            self.rho = torch.cat((self.rho[1:], (1.0 / torch.dot(self.S[-1], self.Y[-1])).view(1, -1)), dim=0)

            self.k += 1
            dfnrm = torch.norm(df).item()
            values = [self.k, f.item(), dfnrm, dfnrm / nrmdf0, torch.norm(p - p_old).item(), alpha, ls_iter, zoom_iter]
            values += obj_fctn.get_metrics(p, x, y, x_val, y_val)
            info['values'] = torch.cat((info['values'], torch.tensor(values).reshape(1, -1)), dim=0)
            print(info['frmt'].format(*values))

            p_old, f_old, df_old = p.clone(), f.clone(), df.clone()

            if alpha == 0:
                print('Linesearch Break')
                break
                # raise ValueError('Linesearch Break')

        # release memory
        self.S = None
        self.Y = None
        self.rho = None
        return p, info

    def two_loop_recursion(self, q):
        alpha = torch.zeros(self.m, device=q.device, dtype=q.dtype)
        for i in range(self.m - 1, -1, -1):
            # newest to oldest
            alpha[i] = self.rho[i] * torch.dot(self.S[i].view(-1), q.view(-1))
            q -= alpha[i] * self.Y[i]

        r = self.gamma * q

        for i in range(self.m):
            # oldest to newest
            beta = self.rho[i] * torch.dot(self.Y[i].view(-1), r.view(-1))
            r += (alpha[i] - beta) * self.S[i]

        return r


class WolfeLineSearch:

    def __init__(self):
        self.max_iter = 50
        self.c1 = 1e-4
        self.c2 = 0.9
        self.alpha_max = 5
        self.max_zoom_iter = 50

    def search(self, obj_fctn, p, d, x, y, alphac=None):
        phi0, dphi0 = self.phi(obj_fctn, p, d, x, y, 0, do_gradient=True)
        alpha_old, phi_old, dphi_old = 0, phi0.clone(), dphi0.clone()

        if alphac is None or alphac == 0:
            alphac = 0.5 * self.alpha_max

        zoom_iter = 0
        iter = 1
        while iter <= self.max_iter:
            phic, dphic = self.phi(obj_fctn, p, d, x, y, alphac, do_gradient=True)

            if (phic > phi0 + self.c1 * alphac * dphi0) or (phic > phi_old and iter > 0):
                print('here1')
                alpha_opt, zoom_iter = self.zoom(obj_fctn, p, d, x, y, phi0, dphi0, alpha_old, phi_old, dphi_old, alphac, phic, dphic)
                if isinstance(alphac, torch.Tensor):
                    alpha_opt = alpha_opt.item()

                return alpha_opt, iter, zoom_iter

            if abs(dphic) <= -self.c2 * dphi0:
                print('here2')
                if isinstance(alphac, torch.Tensor):
                    alpha_opt = alphac.item()
                else:
                    alpha_opt = alphac
                return alpha_opt, iter, zoom_iter

            if dphic >= 0:
                print('here3')
                alpha_opt, zoom_iter = self.zoom(obj_fctn, p, d, x, y, phi0, dphi0, alphac, phic, dphic, alpha_old, phi_old, dphi_old)
                if isinstance(alpha_opt, torch.Tensor):
                    alpha_opt = alpha_opt.item()
                return alpha_opt, iter, zoom_iter

            alpha_old, phi_old, dphi_old = deepcopy(alphac), phic.clone(), dphic.clone()
            alphac = (alphac + self.alpha_max) / 2

            iter += 1

        # did not converge
        return 0, iter, zoom_iter

    def zoom(self, obj_fctn, p, d, x, y, phi0, dphi0, alphaLo, phiLo, dphiLo, alphaHi, phiHi, dphiHi):
        iter = 1
        while iter <= self.max_zoom_iter:
            alpha = self.interpolate(alphaLo, phiLo, dphiLo, alphaHi, phiHi, dphiHi, phi0, dphi0)
            # alpha = 0.5 * (alphaLo + alphaHi)

            phic, dphic = self.phi(obj_fctn, p, d, x, y, alpha, do_gradient=True)
            if phic > phi0 + self.c1 * alpha or phic >= phiLo:
                if isinstance(alpha, torch.Tensor):
                    alpha = alpha.item()

                alphaHi = alpha
            else:
                if abs(dphic) <= -self.c2 * dphi0:
                    if isinstance(alpha, torch.Tensor):
                        alpha = alpha.item()
                    return alpha, iter

                if dphic * (alphaHi - alphaLo) >= 0:
                    alphaHi = alphaLo

                alphaLo = alpha
                phiLo = phic

            iter += 1

        # did not converge
        return 0, iter

    @staticmethod
    def phi(obj_fctn, p, d, x, y, alpha, do_gradient=False):
        phic, dphic = None, None
        if do_gradient:
            phic, dfc = obj_fctn.evaluate(p + alpha * d, x, y, do_gradient=True)
            dphic = torch.dot(dfc.view(-1), d.view(-1))
        else:
            phic, _ = obj_fctn.evaluate(p + alpha * d, x, y, do_gradient=False)

        return phic, dphic

    @staticmethod
    def interpolate(alphaLo, phiLo, dphiLo, alphaHi, phiHi, dphiHi, phi0=None, dphi0=None):
        # alpha = -(alphaLo ** 2 * dphiHi) / (2 * (phiLo - phiHi - alphaLo * dphiHi))
        # alpha = cubicmin(alphaLo, phiLo, dphiLo, alphaHi, phiHi, 0, phi0)[0]
        alpha = quadraticmin(alphaLo, phiLo, dphiLo, alphaHi, phiHi)[0]

        myEps = 0.1 * abs(alphaHi - alphaLo)
        if (alpha is None) or (alpha <= myEps + min(alphaHi, alphaLo)): # or (alpha >= max(alphaHi, alphaLo) - myEps):
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
        xmin = -b / (2 * a)
        fmin = a * xmin ** 2 + b * xmin + c
    else:
        xmin = None
        fmin = None

    return xmin, fmin, (a, b, c)



class ArmijoLineSearch:

    def __init__(self):
        self.max_iter = 20
        self.gamma = 1e-3

    def search(self, obj_fctn, p, d, f, df, x, y, alpha=1.0):

        k = 0
        tau = torch.dot(d.view(-1), df.view(-1))
        while k < self.max_iter:
            f2 = obj_fctn.evaluate(p + alpha * d, x, y)[0]

            if f2 <= f + self.gamma * tau:
                break

            alpha *= 0.5
            k += 1

        return alpha, k
