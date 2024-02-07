import numpy as np
import matplotlib.pyplot as plt

pt = np.loadtxt("pt.txt")
hls = np.loadtxt("hls.txt")



# exp_table = np.loadtxt("exp_table.txt")
# # hls = np.exp(hls)
# # hls = np.clip(hls, 0, 256)
# # hls = np.floor(hls*2**8)/2**8

# for i in range(hls.shape[0]):
# 	elem = hls[i]
# 	if elem <= -8:
# 		hls[i] = 0
# 	elif elem >= 7.93750000:
# 		hls[i] = exp_table[255]
# 	else:
# 		elem -= (-8)
# 		idx = elem*16
# 		idx = int(idx)
# 		hls[i] = exp_table[idx]

# hls = hls.reshape(64,64)
# sums = hls.sum(-1)[...,None]
# hls /= sums
# hls = hls.reshape(-1)
# hls = np.floor(hls*2**7)/2**7


print(pt.min(), pt.max())
print(hls.min(), hls.max())

if (len(pt) != len(hls)):
	raise Exception("Shape not same, {} vs {}".format(pt.shape, hls.shape))
diff = pt-hls
if not np.allclose(pt, hls):
	print("DIFFERENCE")
	print(min(diff), max(diff))
	print(np.count_nonzero(diff))
	print(np.nonzero(diff)[0])
plt.plot(pt, hls, 'bo')
plt.show()
plt.savefig("debug.png")

