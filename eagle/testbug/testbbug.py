import os
import time
import random


import torch
import numpy as np

testindex = "pos_diff"


based = np.zeros(25)
based[8] = 0.9
based[19] = 0.1
draftd = 0.7 * based + 0.3 * np.random.dirichlet(0.3 * np.ones(len(based)))
draftd = draftd / draftd.sum()


maxlen=100
nbased=np.zeros((maxlen,25))
for i in range(maxlen):
    r=random.random()
    index=random.sample(range(25), 2)
    nbased[i,index[0]] = r
    nbased[i,index[1]] = 1-r
    nbased[i] = 0.75 * nbased[i] + 0.25 * np.random.dirichlet(0.3 * np.ones(len(nbased[i])))
    nbased[i] = nbased[i] / nbased[i].sum()

based = torch.as_tensor(nbased)
draftd = torch.as_tensor(draftd)




from model.ea_model import EaModel

model = EaModel.from_pretrained(
    based=based,
    draftd=draftd,

)

outs = []
input_ids = torch.as_tensor([[1, 2]])
s = time.time()
for i in range(500000):
    output_ids = model.eagenerate(input_ids, temperature=1.0, max_new_tokens=15)
    outs.append(output_ids[:, input_ids.shape[1]:input_ids.shape[1] + 15])
    if i>0 and i%1000==0:
        outstensor = torch.cat(outs, dim=0)
        e = time.time()
        print(e - s,i)
        torch.save(outstensor, f"testresult_{testindex}.pt")
        torch.save(based, f"testbased_{testindex}.pt")
        torch.save(draftd, f"testdraftd_{testindex}.pt")
outstensor = torch.cat(outs, dim=0)
e = time.time()
print(e - s)
torch.save(outstensor, f"testresult_{testindex}.pt")
torch.save(based, f"testbased_{testindex}.pt")
torch.save(draftd, f"testdraftd_{testindex}.pt")
