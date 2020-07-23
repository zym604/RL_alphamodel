from btgym import BTgymEnv
env = BTgymEnv(filename='/root/btgym/examples/data/DAT_ASCII_EURUSD_M1_2016.csv') 
o = env.reset()

for i in range(2): 
    action = env.action_space.sample() # random action
    obs, reward, done, info = env.step(action) 
    print('ACTION: {}\nREWARD: {}\nINFO: {}'.format(action, reward, info))
    #print(info[0]['step'],info[0]['broker_value'])
    
env.close()
