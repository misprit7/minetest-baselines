import numpy as np

def relative_entropy(probabilities):
    """
    Calculates the shannon entropy of current action probabilities to maximum
    possible entropies
    For a model, expect to start near 1 and trend toward 0 but never reach exactly 0

    Parameters
    ------
    probabilties: probability of taking each action. Sum of probabilties = 1

    Returns
    -------
    Relative entropy (0 < relative entropy < 1)
    """
    curr_entropy = sum(p * np.log(1 / p) for p in probabilities)
    max_entropy = np.log(probabilities.size)
    return curr_entropy / max_entropy

if __name__ == "__main__":
    exit()