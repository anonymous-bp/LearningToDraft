import torch
import numpy as np
checkpoint = torch.load("models_last/checkpoint")
print(checkpoint)
#print(type(checkpoint))
for k,v in checkpoint.items():
    if k == "network":
        for k1,v1 in v.items():
            for i in list(v1.cpu().numpy()):
                if np.any(i) == np.nan:
                    print(k1)
                if np.any(i) > 1:
                    print(k1)
                if np.any(i)  <= 10e-6:
                    print(k1)
    #print(k,v.numpy().shape())
