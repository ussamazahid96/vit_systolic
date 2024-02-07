import numpy as np

def array2ap_uints(array, precision):
    array = np.where(array < 0, array+(1 << precision), array).astype(np.uint64)
    factor = 1 << precision*np.arange(array.shape[-1], dtype=np.uint64)
    val = array.dot(factor)
    return val


def pack_weights(x, PEs, SIMDs, wprec):
    H, W = x.shape
    if (H%PEs!=0 or W%SIMDs!=0):
        raise Exception("PEs/SIMDs not a multiple of H/W")
    neededWMem = (H * W) // (PEs * SIMDs)

    x = x.reshape(H, -1, SIMDs)
    x = array2ap_uints(x, wprec)

    x = np.split(x, H/PEs, axis=0)
    x = np.asarray(x)
    x = np.split(x, W/SIMDs, axis=-1)
    x = np.asarray(x).swapaxes(0,2)
    x = x.reshape(x.shape[0], -1)  
    return x