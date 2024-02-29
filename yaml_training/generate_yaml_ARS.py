import os
import glob
import random

from server_utils import save_config, get_last_args_id


rng = random.Random()

############################################
# Some auxiliary function
############################################
def permutate_dict(env_config):
    l = 1
    for k, v in env_config.items():
        l *= len(v)

    args = [env_config.copy()]

    i = 0
    j = 0
    for k in env_config.keys():
        for s in range(i, j+1):
            values = args[s][k]
            for v in values:
                tmp_dict = args[i].copy()
                tmp_dict[k] = v
                args.append(tmp_dict)
                j += 1
            i += 1
    return args[-l:]

def list_fusion(list_1, list_2):
    new_list = []
    for el_2 in list_2:
        for el in list_1:
            new_el = el + "," + el_2
            new_list.append(new_el)
    return new_list

##########################
# Parameters for training
##########################
ALGO = 'ars'
ENV = 'QuadrupedSpring-v0'
N_ENVS = 40
N_EVAL_ENVS = 4
N_ENVS = 10
N_EVAL_ENVS = 1
SAVE_FREQ = 5_000_000
EVAL_FREQ = 5_000_000
LOG_INTERVAL = 10
EVAL_EPISODES = 4
N_DELTA = 60
N_TOP = 30
LEARNING_RATE = 0.03
DELTA_STD = 0.025
N_STEP = 75_000_000

# POLICIES = ['LinearPolicy']#, 'MlpPolicy']
POLICIES = ['MlpPolicy']
# NETS = ['dict(net_arch=[16]',
#         'dict(net_arch=[20]',
#         'dict(net_arch=[24]',
#         'dict(net_arch=[64]',
#         ]
# NETS = ['dict(net_arch=[16]',
#         'dict(net_arch=[32]',
#         'dict(net_arch=[64]',
#         'dict(net_arch=[16, 16]',
#         'dict(net_arch=[32, 32]',
#         'dict(net_arch=[64, 64]',
#         ]
NETS = ['dict(net_arch=[64]']
# NETS = ['dict(net_arch=[12]',
#         'dict(net_arch=[20]',
#         'dict(net_arch=[16]',
#         ]
# ACTIVATION_FNS = ['activation_fn=nn.ReLU)', 'activation_fn=nn.Tanh)']
ACTIVATION_FNS = ['activation_fn=nn.Tanh)']

#####################################
# Environment combination parameters 
#####################################
MOTOR_CONTROL_MODE = ['PD']
ACTION_REPEAT = 10
ENABLE_SPRINGS = [True]
ENABLE_ACTION_INTERPOLATION = [False]
ENABLE_ACTION_FILTER = True
TASK_ENV = ['CONTINUOUS_JUMPING_FORWARD3']
OBSERVATION_SPACE_MODE = ['ARS_BASIC']
ACTION_SPACE_MODE = ['SYMMETRIC']
ENV_RANDOMIZER_MODE = 'GROUND_RANDOMIZER'
CURRICULUM = 0.0

if ALGO == 'ars':
    ARS_N_EVAL_EPISODES = 3 #  if ENABLE_ENV_RANDOMIZATION else 1

########################################
# wrappers
########################################
ENABLE_LANDING_WRAPPER = False
ENABLE_LANDNIG_WRAPPPER2 = False
ENABLE_LANDING_CONTINUOUS_WRAPPER = False
ENABLE_LANDING_CONTINUOUS_WRAPPER2 = True
ENABLE_LANDING_WRAPPER_BACKFLIP = False
ENABLE_GO_TO_REST = False

########################################
# path 
########################################
PATH = 'yaml_training/train_env_kwargs'
FOLDER = 'logs'

########################################
# flags
########################################
ENABLE_CURRICULUM = False
REMOVE_PREVIOUS_ARGS = True
DISABLE_WRAPPERS = False


if __name__ == '__main__':

    os.makedirs(PATH, exist_ok=True)

    if REMOVE_PREVIOUS_ARGS:
        for file in glob.glob(os.path.join(PATH, '*')):
            os.remove(file)
        last_id = 0
    else:
        last_id = get_last_args_id()

    training_args = {'algo': ALGO,
                    'env': ENV,
                    '-n': N_STEP,
                    'eval_freq': EVAL_FREQ,
                    'save_freq': SAVE_FREQ,
                    'eval_episodes': EVAL_EPISODES,
                    'n_eval_envs': N_EVAL_ENVS,
                    'log_interval': LOG_INTERVAL,
                    'log_folder': FOLDER,
                    }

    hyperparams = {'n_envs': N_ENVS}
    hyperparams['n_top'] = N_TOP
    hyperparams['n_delta'] = N_DELTA
    hyperparams['delta_std'] = DELTA_STD
    hyperparams['learning_rate'] = LEARNING_RATE


    import_path = 'quadruped_spring.env.wrappers.'
    rest_wrapper = import_path + 'rest_wrapper.RestWrapper'
    landing_wrapper = import_path + 'landing_wrapper.LandingWrapper'
    landing_wrapper_2 = import_path + 'landing_wrapper_2.LandingWrapper2'
    landing_wrapper_backflip = import_path + 'landing_wrapper_backflip.LandingWrapperBackflip'
    obs_flattening_wrapper = import_path + 'obs_flattening_wrapper.ObsFlatteningWrapper'
    go_to_rest_wrapper = import_path + 'go_to_rest_wrapper.GoToRestWrapper'
    continuous_landing_wrapper = import_path + 'landing_wrapper_continuous.LandingWrapperContinuous'
    continuous_landing_wrapper2 = import_path + 'landing_wrapper_continuous2.LandingWrapperContinuous2'

    architectures = list_fusion(NETS, ACTIVATION_FNS)
    id = 0
    for policy in POLICIES:
        if policy == 'MlpPolicy':
            hyperparams['policy'] = policy
            hyperparams['policy_kwargs'] = architectures
        
        env_config = {}
        env_config['motor_control_mode'] = MOTOR_CONTROL_MODE
        env_config['action_repeat'] = ACTION_REPEAT
        env_config['enable_springs'] = ENABLE_SPRINGS
        env_config['enable_action_interpolation'] = ENABLE_ACTION_INTERPOLATION
        env_config['enable_action_filter'] = ENABLE_ACTION_FILTER
        env_config['task_env'] = TASK_ENV
        env_config['observation_space_mode'] = OBSERVATION_SPACE_MODE
        env_config['action_space_mode'] = ACTION_SPACE_MODE
        env_config['env_randomizer_mode'] = ENV_RANDOMIZER_MODE
        env_config['curriculum_level'] = CURRICULUM
        
        aux_config = {}
        for k, v in env_config.items():
            if not isinstance(v, list):
                aux_config[k] = [v]
            else:
                aux_config[k] = v
        possible_env_configs = permutate_dict(aux_config)
        
        aux_config = {}
        for k, v in hyperparams.items():
            if not isinstance(v, list):
                aux_config[k] = [v]
            else:
                aux_config[k] = v
        possible_hyper_params = permutate_dict(aux_config)
        
        for hyperparam in possible_hyper_params:
            wrapper_list = []
            if ENABLE_LANDING_WRAPPER:
                wrapper_list.append(landing_wrapper)
            if ENABLE_LANDNIG_WRAPPPER2:
                wrapper_list.append(landing_wrapper_2)
            if ENABLE_LANDING_WRAPPER_BACKFLIP:
                wrapper_list.append(landing_wrapper_backflip)
            if ENABLE_GO_TO_REST:
                wrapper_list.append(go_to_rest_wrapper)
            if ENABLE_LANDING_CONTINUOUS_WRAPPER:
                wrapper_list.append(continuous_landing_wrapper)
            if ENABLE_LANDING_CONTINUOUS_WRAPPER2:
                wrapper_list.append(continuous_landing_wrapper2)
            
            wrapper_list.append(obs_flattening_wrapper)
            for env_config in possible_env_configs:
                kwargs = training_args.copy()
                kwargs['env_kwargs'] = env_config
                if ALGO == 'ars':
                    kwargs['ars_n_eval_episodes'] = ARS_N_EVAL_EPISODES
                kwargs['curriculum'] = ENABLE_CURRICULUM
                hyperparam['env_wrapper'] = wrapper_list
                kwargs['hyperparams'] = hyperparam
                kwargs['tensorboard_log'] = "tensorlog"
                kwargs['uuid'] = False
                save_config(args=kwargs, id=id+last_id, path=PATH)
                id += 1
    
    print('***********************************')
    print(f'all possible env configs saved in:\n{PATH}')
    print('***********************************')
