import numpy as np

def relative_entropy(probabilities):
    curr_entropy = sum(p * np.log(1 / p) for p in probabilities)
    max_entropy = np.log(probabilities.size)
    return curr_entropy / max_entropy

if __name__ == "__main__":
    exit()