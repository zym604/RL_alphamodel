#_, reward, done, _ = env.step(action.item())
# action就是离散化的insight，在0-100之间
class stockenv:
    def __init__(self):
        self.action_space = [0,10,20,30,40,50,60,70,80,90,100]
        self.action_number = len(self.action_space)
        self.histry_action = []
    def reset(self):
        self.histry_action = []
    def getstate(self):
        #返回state，也就是多个不同alpha model的insights
    def step(self,action):
        self.action = action
        #change file parameters: action, start time, etc
        #让quantconnect按照一个给定的histry_action来运行
        self.histry_action.append(self.action)
        print(self.histry_action)
        reward = 1
        if len(self.histry_action)<4:
            done = False
        else:
            done = True
        return reward, done
    def close(self):
        self.histry_action = []

env = stockenv()
env.reset()
for action in [60,20,70,40,10]:
    reward, done = env.step(action)
    print(action, done)
print(env.action_space)
print(env.action_number)
