import theano
import numpy as np
from pycuda import gpuarray, cumath
from skcuda import linalg, misc, cudart
from theano.tensor import as_tensor_variable
from theano.gof import Op, Apply
import theano.tensor as tt
import pycuda.autoinit
cudart.cudaSetDevice(0)
linalg.init()

class logDect(Op):
    def make_node(self, *inputs):
        alpha = as_tensor_variable(inputs[0])
        xt = as_tensor_variable(inputs[1])
        xf = as_tensor_variable(inputs[2])
        ll = as_tensor_variable(.1)
        return Apply(self, [alpha, xt, xf], [ll.type()])


    def make_thunk(self, node, storage_map, compute_map, rem=None, impl=None, no_recycling=[]):
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]
        A = gpuarray.to_gpu(self.A)
        b = gpuarray.to_gpu(self.b)
        x = gpuarray.to_gpu(self.x)
        depth = gpuarray.to_gpu(self.depth)

        def thunk():
            alpha = gpuarray.to_gpu(np.squeeze(np.asarray(inputs[0]))[:, None])
            x_t = gpuarray.to_gpu(np.asarray(inputs[1])[0, :, :])
            x_f = gpuarray.to_gpu(np.asarray(inputs[2])[0, :, :])
            Xt = cumath.exp(misc.add(linalg.dot(x_t, A), b))
            Xf = cumath.exp(misc.add(linalg.dot(x_f, A), b))
            Xtn = misc.sum(Xt, axis=1, keepdims=True)
            Xfn = misc.sum(Xf, axis=1, keepdims=True)
            Xt = misc.divide(Xt, Xtn)
            Xf = misc.divide(Xf, Xfn)
            w = misc.multiply(Xt, alpha) + misc.multiply(Xf, 1-alpha)
            wp = cumath.log(w)
            wpn = misc.sum(wp, axis=1, keepdims=True) / self.n
            wp = misc.subtract(wp, wpn)
            t1 = misc.sum(x * wp, axis=1)
            t2 = (self.n + depth) * cumath.log(misc.sum(w, axis=1))
            t3 = depth * wpn
            outputs[0][0] = misc.sum(t1 - t2 + t3).get()
            for v in node.outputs:
                compute_map[v][0] = True

        return thunk

    def __init__(self, A, b, n, x):
        self.A = np.asarray(A, dtype=theano.config.floatX)
        self.b =np.asarray(b[True, :], dtype=theano.config.floatX)
        self.n = n
        self.x = np.asarray(x, dtype=theano.config.floatX)
        self.depth = np.sum(x, axis=1)
        self._mgrad = logDectGrad(self)
        super(logDect, self).__init__()

    def grad(self, inputs, g):
        grads = self._mgrad(inputs[0], inputs[1], inputs[2])
        return [g[0] * grads[0], g[0] * grads[1], g[0] * grads[2]]

class logDectGrad(theano.Op):
    def make_node(self, *inputs):
        alpha = as_tensor_variable(inputs[0])
        xt = as_tensor_variable(inputs[1])
        xf = as_tensor_variable(inputs[2])
        return Apply(self, [alpha, xt, xf], [alpha.type(), xt.type(), xf.type()])

    def make_thunk(self, node, storage_map, compute_map, rem=None, impl=None, no_recycling=[]):
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]
        A = gpuarray.to_gpu(self.base.A)
        b = gpuarray.to_gpu(self.base.b)
        x = gpuarray.to_gpu(self.base.x)
        depth = gpuarray.to_gpu(self.base.depth)

        def thunk():
            alpha = gpuarray.to_gpu(np.squeeze(np.asarray(inputs[0]))[:, None])
            x_t = gpuarray.to_gpu(np.asarray(inputs[1])[0, :, :])
            x_f = gpuarray.to_gpu(np.asarray(inputs[2])[0, :, :])
            Xt = cumath.exp(misc.add(linalg.dot(x_t, A), b))
            Xf = cumath.exp(misc.add(linalg.dot(x_f, A), b))
            Xtn = misc.sum(Xt, axis=1, keepdims=True)
            Xfn = misc.sum(Xf, axis=1, keepdims=True)
            Xt = misc.divide(Xt, Xtn)
            Xf = misc.divide(Xf, Xfn)
            w = misc.multiply(Xt, alpha) + misc.multiply(Xf, 1-alpha)
            dq = Xt - Xf
            qdw = dq / w
            t1 = misc.sum(x * qdw, axis=1)
            f = 2 * depth + self.base.n
            t2 = f * misc.sum(dq, axis=1) / misc.sum(w, axis=1)
            t3 = misc.sum(x, axis=1) * misc.sum(qdw, axis=1)
            dalpha = t1 - t2 + t3
            del dq,t1,f,t2,t3

            iw = 1/w
            S1 = misc.multiply(depth[:, None] * (self.base.n-1) / self.base.n, iw)
            S2 = (self.base.n + depth[:, None]) / cumath.log(misc.sum(w, axis=1, keepdims=True))
            F = misc.multiply(misc.subtract((x * iw) - S1, S2), alpha)
            del w,iw,S1,S2

            cast = gpuarray.zeros((x_t.shape[1], Xt.shape[1]), dtype=theano.config.floatX)
            dLq_t = gpuarray.zeros(x_t.shape, dtype=theano.config.floatX)
            dLq_f = gpuarray.zeros(x_f.shape, dtype=theano.config.floatX)
            for i in range(Xt.shape[0]):
                S1 = misc.multiply(Xt[None, i, :], A)
                S2 = misc.sum(S1, axis=1, keepdims=True)
                S2 = misc.multiply(S2, misc.add(Xt[None, i, :], cast))
                dLq_t[i, :] = misc.sum(misc.multiply(F[None, i, :], S1 - S2), axis=1)
                S1 = misc.multiply(Xf[None, i, :], A)
                S2 = misc.sum(S1, axis=1, keepdims=True)
                S2 = misc.multiply(S2, misc.add(Xf[None, i, :], cast))
                dLq_f[i, :] = misc.sum(misc.multiply(F[None, i, :], S1 - S2), axis=1)
            outputs[0][0] = dalpha.get()
            outputs[1][0] = dLq_t.get()
            outputs[2][0] = dLq_f.get()
            for v in node.outputs:
                compute_map[v][0] = True

        return thunk

    def __init__(self, base):
        self.base = base
        super(logDectGrad, self).__init__()
