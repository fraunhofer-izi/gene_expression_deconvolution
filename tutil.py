"""
Some theano utilities.
"""
import os

def setFlag(flag, value = None):
    """
    Description
    -----------
    Sets or overites the theano `flag` in the envirnment variable 'THEANO_FLAGS'.

    Parameter
    ---------
    flag : The flag name that is to be overwritten or set.
    value : The value to be asigned to the flag. If it is
    `None` then `flag` will be pasted as is into 'THEANO_FLAGS'.

    Value
    -----
    The new value of 'THEANO_FLAGS'.
    """
    if (not isinstance(flag, str)):
        raise TypeError("The arrgument `flag` needs to be a string.")
    if ('THEANO_FLAGS' in os.environ):
        flagString = os.getenv('THEANO_FLAGS')
    else:
        flagString = ''

    if (value is None):
        newFlagString = flagString + "," + flag
        os.environ['THEANO_FLAGS'] = newFlagString
        return newFlagString

    if (not isinstance(value, str)):
        raise TypeError("The arrgument `value` needs to be a string or `None`.")

    oldFlags = flagString.split(',')
    flagTag = flag + '='
    newFlags = [s for s in oldFlags if flagTag not in s]
    newFlags.append(flagTag + value)
    newFlagString = ','.join(newFlags)
    os.environ['THEANO_FLAGS'] = newFlagString
    return newFlagString

def runTest(iters = 1000):
    """
    Runs theano matrix exponential as benchmark `iter` times.
    The needed time in seconds will be returned and some
    details will be printed out.
    """
    import numpy
    import time
    import theano

    vlen = 10 * 30 * 768  # 10 x #cores x # threads per core

    rng = numpy.random.RandomState(22)
    x = theano.shared(numpy.asarray(rng.rand(vlen), theano.config.floatX))
    f = theano.function([], theano.tensor.exp(x))
    print(f.maker.fgraph.toposort())
    r = f() # warmup
    t0 = time.time()
    for i in range(iters):
        r = f()
    t1 = time.time()
    tDiff = t1 - t0
    print("Looping %d times took %f seconds" % (iters, tDiff))
    print("Result is %s" % (r,))
    if numpy.any([isinstance(x.op, theano.tensor.Elemwise) and
                  ('Gpu' not in type(x.op).__name__)
                  for x in f.maker.fgraph.toposort()]):
        print('Used the cpu')
    else:
        print('Used the gpu')
    return tDiff
