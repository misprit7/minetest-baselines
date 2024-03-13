import random

class Curriculum():
    def __init__(self, env_dirs):
        self.env_index = 0
        self.env_dirs = env_dirs
    
    def suggest_env():
        pass

class UniformCurriculum(Curriculum):
    def suggest_env(self):
        self.env_index = random.randint(len(self.env_dirs))
        return self.env_dirs[self.env_index]
    
class SequentialCurriculum(Curriculum):
    # num_resets: the number of resets (episodes) between environment switches
    def __init__(self, env_dirs, num_resets = 1):
        self.num_resets = num_resets
        self.reset_counter = 0
        super().__init__(env_dirs)

    def suggest_env(self):
        choice = self.env_dirs[self.env_index]
        self.reset_counter += 1
        if self.reset_counter == self.num_resets:
            self.env_index = (self.env_index + 1) % len(self.env_dirs)
            self.reset_counter = 0
        return choice
    
# class OptimizedCurriculum(Curriculum):
#     def __init__(self, env_dirs, k):
#         self.k = k
#         self.last_reward = [0 for i in env_dirs]
#         self.gradients = [0 for i in env_dirs]
#         self.probability = [1 / len(env_dirs) for i in env_dirs]
#         super().__init__(env_dirs)

#     def update_gradient(self, reward):
#         self.gradients[self.curr_index] += (reward - self.last_reward[self.curr_index]) 

#     def update_probability(self):
#         self.gradients /= self.k

#         self.probability = 

#     def suggest_env(self):
