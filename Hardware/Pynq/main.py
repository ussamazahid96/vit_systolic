import os
import time
import pynq
import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if os.environ['BOARD'] == 'Ultra96':
    PLATFORM="ultra96"
elif os.environ['BOARD'] == 'ZCU104':
    PLATFORM="zcu104"
else:
    raise RuntimeError("Board not supported")


DTYPE = {
    8: np.int8,
    16: np.int16,
    32: np.int32,
    64: np.int64
}

def quantize_and_pack(x, bw,ibw):
    x = np.round(x*2**(bw-ibw))
    x = x.astype(DTYPE[bw]).view(np.uint64)
    return x


def unpack(x, bw,ibw):
    x = np.copy(np.frombuffer(x, np.uint64))
    x = x.view(DTYPE[bw])/2**(bw-ibw)
    return x


class TransAccl:
    def __init__(self, path):
        ol = pynq.Overlay(path)
        
        self.rdma = ol.TransRDMA_0.register_map
        self.accl = ol.AttnHead_0.register_map
        self.accl1 = ol.AttnHead_1.register_map
        self.headfc = ol.HeadFC_0.register_map

    def __call__(self, inputs, qhfc_weights_buffer, key_weights_buffer, value_weights_buffer, output, MemInit):
        
        self.rdma.input_and_weights_1 = inputs.physical_address & 0xffffffff
        self.rdma.input_and_weights_2 = (inputs.physical_address >> 32) & 0xffffffff

        self.rdma.qhfc_weights_1 = qhfc_weights_buffer.physical_address & 0xffffffff
        self.rdma.qhfc_weights_2 = (qhfc_weights_buffer.physical_address >> 32) & 0xffffffff

        self.rdma.key_weights_1 = key_weights_buffer.physical_address & 0xffffffff
        self.rdma.key_weights_2 = (key_weights_buffer.physical_address >> 32) & 0xffffffff

        self.rdma.value_weights_1 = value_weights_buffer.physical_address & 0xffffffff
        self.rdma.value_weights_2 = (value_weights_buffer.physical_address >> 32) & 0xffffffff

        self.headfc.output_r_1 = output.physical_address & 0xffffffff
        self.headfc.output_r_2 = (output.physical_address >> 32) & 0xffffffff

        self.rdma.MemInit = MemInit
        self.accl.MemInit = MemInit
        self.accl1.MemInit = MemInit
        self.headfc.MemInit = MemInit
        
        self.headfc.CTRL.AP_START = 1
        self.accl.CTRL.AP_START = 1
        self.accl1.CTRL.AP_START = 1
        self.rdma.CTRL.AP_START = 1
        # while not (self.headfc.CTRL.AP_DONE):
        #     print(self.headfc.CTRL.AP_DONE, end='\r')
        #     pass
        time.sleep(0.5)


if __name__ == '__main__':
    inputs = np.load("../../Training/export/vit/input.npy").reshape(-1)
    input_norm = np.load("../../Training/export/vit/input_norm.npy").reshape(-1)
    qw = np.load("../../Training/export/vit/qhfc_weights.npy").reshape(-1)
    kw = np.load("../../Training/export/vit/key_weights.npy").reshape(-1)
    vw = np.load("../../Training/export/vit/value_weights.npy").reshape(-1)
    inputs = quantize_and_pack(inputs,8,5)
    input_norm = quantize_and_pack(input_norm,8,3)
    inputs = np.concatenate((input_norm, inputs))

    output_shape = (64*512*8//64,)

    input_buffer         = pynq.allocate(inputs.shape, dtype=np.uint64)
    output_buffer        = pynq.allocate(output_shape, dtype=np.uint64)
    qhfc_weights_buffer  = pynq.allocate(qw.shape, dtype=np.uint64)
    key_weights_buffer   = pynq.allocate(kw.shape, dtype=np.uint64)
    value_weights_buffer = pynq.allocate(vw.shape, dtype=np.uint64)
    
    input_buffer[:] = inputs
    qhfc_weights_buffer[:] = qw
    key_weights_buffer[:] = kw
    value_weights_buffer[:] = vw

    acclerator = TransAccl("Bitstreams/vit-{}.bit".format(PLATFORM))
    start = time.time()
    print("Loading Weights...")
    acclerator(input_buffer, qhfc_weights_buffer, key_weights_buffer, value_weights_buffer, output_buffer, 1)
    print("Running Inference...")
    acclerator(input_buffer, qhfc_weights_buffer, key_weights_buffer, value_weights_buffer, output_buffer, 0)
    end = time.time()
    print("Org FPS = {:.4f}".format(1./(end-start)))

    output = unpack(output_buffer,8,4)
    
    np.savetxt("../../debug/pynq.txt", output, fmt='%.8f')
    gref = np.loadtxt("../../debug/pt.txt")
    plt.plot(gref, output, 'bo')
    plt.savefig("../../debug/pynq.png")
