import theano
from theano.tensor import as_tensor_variable
from theano.gof import Op, Apply
import theano.tensor as tt
import numpy as np

class logDect(Op):
    def make_node(self, *inputs):
        alpha = as_tensor_variable(inputs[0])
        xt = as_tensor_variable(inputs[1])
        xf = as_tensor_variable(inputs[2])
        ll = as_tensor_variable(.1)
        return Apply(self, [alpha, xt, xf], [ll.type()])

    def perform(self, node, inputs, output):
        alpha = inputs[0][:, None]
        x_t = inputs[1]
        x_f = inputs[2]
        np.matmul(x_t, self.A, out=self.Xt)
        np.matmul(x_f, self.A, out=self.Xf)
        np.add(self.Xt, self.b, out=self.Xt)
        np.add(self.Xf, self.b, out=self.Xf)
        np.exp(self.Xt, out=self.Xt)
        np.exp(self.Xf, out=self.Xf)
        Xtn = np.sum(self.Xt, axis=1, keepdims=True)
        Xfn = np.sum(self.Xf, axis=1, keepdims=True)
        np.divide(self.Xt, Xtn, out=self.Xt)
        np.divide(self.Xf, Xfn, out=self.Xf)
        self.w = np.multiply(self.Xt, alpha) + np.multiply(self.Xf, 1-alpha)
        wp = np.log(self.w)
        wpn = np.sum(wp, axis=1, keepdims=True) / self.n
        np.subtract(wp, wpn, out=wp)
        t1 = np.sum(self.x * wp, axis=1)
        #t2 = (self.n + self.depth) * np.log(np.sum(self.w, axis=1))
        t3 = self.depth * wpn
        output[0][0] = np.sum(t1 + t3) # - t2 

    def __init__(self, A, b, n, x):
        self.A = np.asarray(A)
        self.b = np.asarray(b[True, :])
        self.n = n
        self.x = np.asarray(x)
        self.depth = np.sum(self.x, axis=1)
        self.nsamp = self.x.shape[0]
        self.ncomp = self.A.shape[0]
        self.ngenes = self.A.shape[1]
        self.c = ((self.n-1)/self.n) * self.depth[:, None] + self.x
        self.Xt = np.zeros((self.nsamp, self.ngenes))
        self.Xf = np.zeros((self.nsamp, self.ngenes))
        self.w = np.zeros((self.nsamp, self.ngenes))
        self._mgrad = logDectGrad(self)
        super(logDect, self).__init__()

    def grad(self, inputs, g):
        grads = self._mgrad(inputs[0], inputs[1], inputs[2])
        return [g[0] * grads[0], g[0] * grads[1], g[0] * grads[2]]
    
class logDectGrad(Op):
    def make_node(self, *inputs):
        alpha = as_tensor_variable(inputs[0])
        xt = as_tensor_variable(inputs[1])
        xf = as_tensor_variable(inputs[2])
        return Apply(self, [alpha, xt, xf], [alpha.type(), xt.type(), xf.type()])

    def perform(self, node, inputs, outputs):
        alpha = inputs[0][:, None]
        x_t = inputs[1]
        x_f = inputs[2]
        dq = self.base.Xt - self.base.Xf
        F = np.divide(self.base.c, self.base.w)
        dalpha = np.sum(F * dq, axis = 1)

        dLq_t = np.zeros(x_t.shape)
        dLq_f = np.zeros(x_f.shape)
        F_t = np.multiply(F, alpha)
        F_f = np.multiply(F, 1-alpha)
        for i in range(self.base.Xt.shape[0]):
            S1 = np.multiply(self.base.Xt[None, i, :], self.base.A)
            S2 = np.sum(S1, axis=1, keepdims=True)
            S2 = np.multiply(S2, self.base.Xt[None, i, :])
            dLq_t[i, :] = np.squeeze(np.matmul(S1 - S2, F_t[i, :, None]))
            S1 = np.multiply(self.base.Xf[None, i, :], self.base.A)
            S2 = np.sum(S1, axis=1, keepdims=True)
            S2 = np.multiply(S2, self.base.Xf[None, i, :])
            dLq_f[i, :] = np.squeeze(np.matmul(S1 - S2, F_f[i, :, None]))
        outputs[0][0] = dalpha
        outputs[1][0] = dLq_t
        outputs[2][0] = dLq_f
        return

    def __init__(self, base):
        self.base = base
        super(logDectGrad, self).__init__()
