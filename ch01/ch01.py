# %%

from bandit import Bandit
import numpy as np
# %%

bandit = Bandit()

Q = 0

ntrial = 100000000

print(f"Rate: {bandit.rates[0]}")

for n in range(1,ntrial):
    rw = bandit.play(0)
    
    Q += (rw - Q) / n
    

print(Q)
# %%
ts = np.zeros(5)

print(ts)
len(ts)
# %%
