import math
import numpy as np

def Solution_(k, t, N0, B):
    # viable for constant B
    t = t * B * N0 / 2
    if t == 0:
        y = math.pow(t, k - 1) * math.pow(1 + t, - k - 1)
    else:
        y = math.pow(t / (t + 1), k + 1) / math.pow(t, 2)  
    return y*N0

Solution = np.vectorize(Solution_)

#Solution([1,2,3],[1,2,3],100,1)
