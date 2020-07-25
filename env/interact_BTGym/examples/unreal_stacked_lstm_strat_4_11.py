import warnings
warnings.filterwarnings("ignore") # suppress h5py deprecation warning

import os
import backtrader as bt
import numpy as np

from btgym import BTgymEnv, BTgymDataset
from btgym.strategy.observers import Reward, Position, NormPnL
from btgym.algorithms import Launcher, Unreal, AacStackedRL2Policy
from btgym.research.strategy_gen_4 import DevStrat_4_11

MyCerebro = bt.Cerebro()

# Define strategy and broker account parameters:
MyCerebro.addstrategy(
    DevStrat_4_11,
    start_cash=2000,  # initial broker cash
    commission=0.0001,  # commisssion to imitate spread
    leverage=10.0,
    order_size=2000,  # fixed stake, mind leverage
    drawdown_call=10, # max % to loose, in percent of initial cash
    target_call=10,  # max % to win, same
    skip_frame=10,
    gamma=0.99,
    reward_scale=7, # gardient`s nitrox, touch with care!
    state_ext_scale = np.linspace(3e3, 1e3, num=5)
)
# Visualisations for reward, position and PnL dynamics:
MyCerebro.addobserver(Reward)
MyCerebro.addobserver(Position)
MyCerebro.addobserver(NormPnL)

# Data: uncomment to get up to six month of 1 minute bars:
data_m1_6_month = [
    './data/DAT_ASCII_EURUSD_M1_201701.csv',
    './data/DAT_ASCII_EURUSD_M1_201702.csv',
    './data/DAT_ASCII_EURUSD_M1_201703.csv',
    './data/DAT_ASCII_EURUSD_M1_201704.csv',
    './data/DAT_ASCII_EURUSD_M1_201705.csv',
    './data/DAT_ASCII_EURUSD_M1_201706.csv',
]

# Uncomment single choice:
MyDataset = BTgymDataset(
    #filename=data_m1_6_month,
    filename='./data/test_sine_1min_period256_delta0002.csv',  # simple sine 
    start_weekdays={0, 1, 2, 3, 4, 5, 6},
    episode_duration={'days': 1, 'hours': 23, 'minutes': 40}, # note: 2day-long episode
    start_00=False,
    time_gap={'hours': 10},
)

env_config = dict(
    class_ref=BTgymEnv, 
    kwargs=dict(
        dataset=MyDataset,
        engine=MyCerebro,
        render_modes=['episode', 'human', 'internal', ], #'external'],
        render_state_as_image=True,
        render_ylabel='OHL_diff. / Internals',
        render_size_episode=(12,8),
        render_size_human=(9, 4),
        render_size_state=(11, 3),
        render_dpi=75,
        port=5000,
        data_port=4999,
        connect_timeout=90,
        verbose=0,
    )
)

cluster_config = dict(
    host='127.0.0.1',
    port=12230,
    num_workers=4,  # set according CPU's available or so
    num_ps=1,
    num_envs=1,
    log_dir=os.path.expanduser('~/tmp/test_4_11'),  # current checkpoints and summaries are here
    initial_ckpt_dir=os.path.expanduser('~/tmp/pre_trained_model/test_4_11'),  # load pre-trained model, if chekpoint found  
)

policy_config = dict(
    class_ref=AacStackedRL2Policy,
    kwargs={
        'lstm_layers': (256, 256),
        'lstm_2_init_period': 60,
    }
)

trainer_config = dict(
    class_ref=Unreal,
    kwargs=dict(
        opt_learn_rate=[1e-4, 1e-4], # random log-uniform 
        opt_end_learn_rate=1e-5,
        opt_decay_steps=50*10**6,
        model_gamma=0.99,
        model_gae_lambda=1.0,
        model_beta=0.05, # entropy reg
        rollout_length=20,
        time_flat=True, 
        use_value_replay=False, 
        model_summary_freq=10,
        episode_summary_freq=1,
        env_render_freq=2,
    )
)

launcher = Launcher(
    cluster_config=cluster_config,
    env_config=env_config,
    trainer_config=trainer_config,
    policy_config=policy_config,
    test_mode=False,
    max_env_steps=100*10**6,
    save_secs=300,  # save checkpoint every N seconds (default is 600)
    root_random_seed=0,
    verbose=0
)

# Train it:
launcher.run()

launcher.export_checkpoint(os.path.expanduser('~/tmp/pre_trained_model/test_4_11'))
