import random

class Curriculum():
    def __init__(self, env_dirs):
        self.env_dirs = env_dirs
    
    def suggest_env():
        pass

class UniformCurriculum(Curriculum):
    def suggest_env(self):
        choice = random.choice(self.env_dirs)
        print(choice)
        return choice
    
class SequentialCurriculum(Curriculum):
    def __init__(self, env_dirs):
        self.curr_index = 0
        super().__init__(env_dirs)

    def suggest_env(self):
        choice = self.env_dirs[self.curr_index]
        print(choice)
        self.curr_index = (self.curr_index + 1) % len(self.env_dirs)
        return choice
    
class OptimizedCurriculum(Curriculum):
    def __init__(self, env_dirs):
        self.gradients = [0 for i in env_dirs]
        self.probability = [1 / len(env_dirs) for i in env_dirs]
        super().__init__(env_dirs)

    def update_probability():
        

    def suggest_env(self):
