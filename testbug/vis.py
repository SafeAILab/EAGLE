import numpy as np
import matplotlib.pyplot as plt
import torch

#6 chain
testindex="pos_diff"
p=torch.load(f"testbased_{testindex}.pt").numpy()
ds=torch.load(f"testresult_{testindex}.pt")
bias=1
pos=ds.shape[1]

for i in range(pos):
    d=ds[:,i].numpy()
    pos_p=p[i+bias]
    plt.hist(d, bins=np.arange(len(pos_p)+1) - 0.5, density=True, alpha=0.5, label='d')
    plt.scatter(np.arange(len(pos_p)), pos_p, label='p', color='red', marker='o')
    plt.title(f"Position {i}")
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.savefig(f"testresult_{testindex}_{i}.png")
    plt.legend()
    plt.show()

