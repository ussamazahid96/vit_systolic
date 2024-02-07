import numpy as np
import math

INT_BITS = 4
FRAC_BITS = 4

O_INT_BITS = 10
O_FRAC_BITS = 8


lower_limit = -2**(INT_BITS-1)
upper_limit =  2**(INT_BITS-1) - 2**-(FRAC_BITS)
NUM_ELEMS = 2**(INT_BITS+FRAC_BITS)

x = np.linspace(lower_limit, upper_limit, NUM_ELEMS)
y = np.exp(x)
y = np.clip(y, 0, 2**(O_INT_BITS))
y = np.floor(y*2**(O_FRAC_BITS))/2**(O_FRAC_BITS)

recip_step = (NUM_ELEMS)/(upper_limit - lower_limit)
bw = math.ceil(math.log2(recip_step))

with open("../Hardware/HLS/tpu/exp_lookup_table.hpp", 'w') as f:
	f.write("#ifndef __EXP_LOOKUP_TABLE_HPP__\n")
	f.write("#define __EXP_LOOKUP_TABLE_HPP__\n")

	f.write("const double lower_limit = {:.8f};\n".format(lower_limit))
	f.write("const double upper_limit = {:.8f};\n".format(upper_limit))
	f.write("const unsigned int NUM_ELEMS = {};\n".format(NUM_ELEMS))

	f.write("const ap_ufixed<{},{}> recip_step = {:.8f};\n".format(bw,bw,recip_step))
	f.write("const ap_ufixed<{}, {}, AP_RND, AP_SAT> exp_lookup_table[{}] = {} \n".format(O_INT_BITS+O_FRAC_BITS, O_INT_BITS, 2**(INT_BITS+FRAC_BITS), "{"))

	for i in range(y.shape[0]):
		endline_char = " " if i == y.shape[0]-1 else ","
		f.write("{:.8f}{}\n".format(y[i], endline_char))

	f.write("};\n")
	f.write("#endif")
np.savetxt("exp_table.txt", y.reshape(-1), fmt='%.8f')