import numpy as np

# Q1

def entropy(probs):
    e = probs * np.log2(probs)
    return -e.sum()

N_num = 10  # number of numerals
N_vow = 5  # number of vowels
N_con = 21  # number of consonants

p = []
for N in [N_num, N_vow, N_con]:
    p += [1/N] * N

p = 1 / 3 * np.array(p)
e = entropy(p)

print(2**e)
