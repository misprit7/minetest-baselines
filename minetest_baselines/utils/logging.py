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

    # Prevent NaN scenarios
    for i in range(len(probabilities)):
        if (probabilities[i] == 0):
            probabilities = probabilities.at[i].set(1e-12)
    
    # Calculations
    curr_entropy = sum(p * np.log(1 / p) for p in probabilities)
    max_entropy = np.log(probabilities.size)
    return curr_entropy / max_entropy

def kl_divergence(old_policy, new_policy):
    """
    Calculates the KL divergence between 2 policies. 

    Large means agent is learning from very different experience from policy. Expected
    if there is a replay buffer, so look for stability. For those without a replay buffer,
    expect a small divergence. If it's very low, then changes are small so can turn up
    learning rate. If growing, then old experiences are getting fed in again so check
    buffering system.

    Parameters
    ------
    old policy: probability of taking each action before model update. 
                Sum of probabilities = 1
    new_policy: probability of taking each action after a model update
                Sum of probabilities = 1

    Returns
    -------
    Kullback-Leibler divergence
    """

    if len(old_policy) != len(new_policy):
        return np.nan
    
    kl_divergence = 0
    for i in range(len(old_policy)):
        # Prevent NaN scenarios
        if old_policy[i] == 0:
            old_policy = old_policy.at[i].set(1e-12)
        if new_policy[i] == 0:
            new_policy = new_policy.at[i].set(1e-12)
        
        kl_divergence += new_policy[i] * np.log(new_policy[i] / old_policy[i])

    return kl_divergence

def action_types(action_log: list):
    total_actions = len(action_log)
    
    look_up = 0
    look_forward = 0
    look_down = 0
    forward = 0
    jump = 0

    for a in range(np.size(action_log)):
        action = action_log[a]

        if (action % 9) % 3 == 0:
            look_up += 1
        elif (action % 9) % 3 == 1:
            look_forward += 1
        else:
            look_down += 1

        if (action % 18) > 8:
            jump += 1
        
        if action >= 18:
            forward += 1
    
    return forward / total_actions, jump / total_actions, {"up": look_up,
                                                           "forward": look_forward,
                                                           "down": look_down}


if __name__ == "__main__":
    exit()
