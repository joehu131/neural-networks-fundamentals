import numpy as np
from matplotlib import pyplot as plt

def getpolicy(Q):
    """ 
    Get best policy matrix from the Q-matrix.
    The policy is simply the index of the action that yields the highest Q-value 
    for each state. 
    """
    
    # np.argmax finds the index of the maximum value. 
    # axis=-1 --> very last dimension (the actions).
    P = np.argmax(Q, axis=-1)

    return P


def getvalue(Q):
    """ 
    Get best value matrix from the Q-matrix.
    The value of a state is the highest Q-value achievable from that state.
    """
    
    # np.max finds the actual maximum value.
    V = np.max(Q, axis=-1)

    return V