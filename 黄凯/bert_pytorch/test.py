import random

aa = [1, 2, 3, 4, 5, 6]
bb = [1, 2, 3, 4, 5, 6]

cc = list(zip(aa, bb))
random.shuffle(cc)
aa[:], bb[:] = zip(*cc)
print(aa, bb)
