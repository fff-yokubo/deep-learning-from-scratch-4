import numpy as np
import time

# naive implementation

def naiveAvg(nmax):
    np.random.seed(0)
    rewards = 0
    for n in range(1, nmax):
        rewards += np.random.rand()
    return rewards / nmax

def naiveAvg2(nmax):
    
    np.random.seed(0)
    
    return sum(np.random.rand() for _ in range(nmax - 1)) / nmax

# incremental implementation

def incrementalAvg(nmax):
    np.random.seed(0)
    Q = 0
    for n in range(1, nmax):
        reward = np.random.rand()
        Q += (reward - Q) / n
    return Q

if __name__ == '__main__':
    
    nmax = 10000000
    
    st = time.time()
    
    out = naiveAvg2(nmax)
    
    print(f"Naive: \t\tQ={out},\t Time = {"%.3f"%(time.time()-st)}")
    
    st = time.time()
    
    out = naiveAvg(nmax)
    
    print(f"Incremental: \tQ={out},\t Time = {"%.3f"%(time.time()-st)}")
    