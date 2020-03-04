#_, reward, done, _ = env.step(action.item())
# action就是离散化的insight，在0-100之间
class stockenv:
    def __init__(self):
        self.action_space = [0,10,20,30,40,50,60,70,80,90,100]
        self.action_number = len(self.action_space)
        self.histry_action = []
        self.root = r'E:\Lean-master'
        self.folder = self.root + r'\Launcher\bin\Debug'
        self.path = self.folder + r'\QuantConnect.Lean.Launcher.exe'
        self.workspace = r'E:\Google\stock'
    def reset(self):
        print("start reset ...")
        self.histry_action = []
        import shutil 
        source = self.workspace + r'\RL\quantconnect_setup_files\config.json'
        destination = self.root + r'\Launcher\config.json'
        shutil.copyfile(source, destination)
        source2 = self.workspace + r'\RL\quantconnect_setup_files\BasicTemplateAlgorithm.py'
        destination2 = self.root + r'\Algorithm.Python\BasicTemplateAlgorithm.py'
        shutil.copyfile(source2, destination2)
        import subprocess
        p1 = subprocess.Popen(self.workspace + r'\interact_lLean\1build_the_project.bat',stdin=subprocess.PIPE,stdout=subprocess.PIPE, shell=True)
        p1.communicate()[0]
        print("finish reset ...")
    def getexeoutput(self):
        import subprocess
        p1 = subprocess.Popen(self.path,cwd=self.folder,stdin=subprocess.PIPE,stdout=subprocess.PIPE, shell=True)
        self.output = p1.communicate()[0]
        return self.output
    def getstate(self):
        #返回state，也就是多个不同alpha model的insights,最好不要读写文件
        self.getexeoutput()
        state =  self.output.splitlines()
        state = [s for s in state if s.startswith(b'STATISTICS')]
        return state
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
        # state = 让quantconnect返回
        state = 0
        return state,reward, done
    def close(self):
        self.histry_action = []

env = stockenv()
env.reset()
state = env.getstate()

print(state[0])
print(state[3])

# for action in [60,20,70,40,10]:
#     state, reward, done = env.step(action)
#     print(action, done)
# print(env.action_space)
# print(env.action_number)
