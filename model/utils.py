from inspect import isfunction

def extract(a, t, x_shape, per_frame=False):
    if per_frame:
        b, c, f, *_ = x_shape
        out = a.gather(0, t.unsqueeze(1).tile([1, f]))
        return out.reshape(b, 1, f, 1, 1)
    else:
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
