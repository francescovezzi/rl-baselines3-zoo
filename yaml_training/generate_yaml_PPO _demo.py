import os
import glob
import inspect

currentfile = inspect.getfile(inspect.currentframe())
rl_zoo_dir = os.path.dirname(os.path.dirname(currentfile))

from server_utils import save_config, get_last_args_id


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

def generate_architectures(list_1, list_2):
    new_list = []
    for el_2 in list_2:
        for el in list_1:
            new_el = "dict(ortho_init=False,"
            new_el += el + "," + el_2
            new_list.append(new_el)
    return new_list

def get_agent_path_id(s):
    return (s.split('/')[-2])[-1]

##########################
# Parameters for training
##########################
ALGO = 'ppo'
ENV = 'QuadrupedSpring-v0'
N_ENVS = 4
N_EVAL_ENVS = 2
SAVE_FREQ = 1_000_000
EVAL_FREQ = 1_000_000
LOG_INTERVAL = 10
EVAL_EPISODES = 10
N_STEPS = 512
N_STEP = 10_000_000
GAMMA = 0.9
N_EPOCHS = 10
BATCH_SIZE = 128  # 64

ACTIVATION_FNS = ['activation_fn=nn.Tanh)']
NETS = ['net_arch=[dict(pi=[64, 64], vf=[64, 64])]']
ARCH = "dict(ortho_init=False," + NETS[0]+ ',' + ACTIVATION_FNS[0]

#####################################
# Environment combination parameters 
#####################################
MOTOR_CONTROL_MODE = ['PD']
ACTION_REPEAT = 10
ENABLE_SPRINGS = [True]
ENABLE_ACTION_INTERPOLATION = [False]
ENABLE_ACTION_FILTER = False
TASK_ENV = ['CONTINUOUS_JUMPING_FORWARD_DEMO']

OBSERVATION_SPACE_MODE = ['PPO_CONTINUOUS_JUMPING_FORWARD']
ACTION_SPACE_MODE = ['SYMMETRIC']
ENV_RANDOMIZER_MODE = ['GROUND_RANDOMIZER']
CURRICULUM = 0.0

########################################
# wrappers
########################################
ENABLE_LANDING_WRAPPER = False
ENABLE_LANDNIG_WRAPPPER2 = False
ENABLE_GO_TO_REST = False
ENABLE_DEMO_WRAPPER = True

########################################
# retrain agent path
########################################
PATH = 'yaml_training/train_env_kwargs'
FOLDER = 'logs'
MODEL = 'best_model.zip'
RETRAIN_PREV_AGENT = False
AGENT_PATH = "path to retrained agent"

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
    
    retrain_paths = []
    for ag_path in AGENT_PATH:
        for file in glob.glob(os.path.join(ag_path, '*')):
            f = os.path.join(file, MODEL)
            retrain_paths.append(f)

    training_args = {'algo': ALGO,
                    'env': ENV,
                    'n_timesteps': N_STEP,
                    'eval_freq': EVAL_FREQ,
                    'save_freq': SAVE_FREQ,
                    'eval_episodes': EVAL_EPISODES,
                    'n_eval_envs': N_EVAL_ENVS,
                    'log_interval': LOG_INTERVAL,
                    'log_folder': FOLDER,
                    }
    
    import_path = 'quadruped_spring.env.wrappers.'
    rest_wrapper = import_path + 'rest_wrapper.RestWrapper'
    landing_wrapper = import_path + 'landing_wrapper.LandingWrapper'
    landing_wrapper_2 = import_path + 'landing_wrapper_2.LandingWrapper2'
    obs_flattening_wrapper = import_path + 'obs_flattening_wrapper.ObsFlatteningWrapper'
    go_to_rest_wrapper = import_path + 'go_to_rest_wrapper.GoToRestWrapper'
    demo_wrapper = import_path + 'reference_state_initialization_wrapper.ReferenceStateInitializationWrapper'
    
    architectures = generate_architectures(NETS, ACTIVATION_FNS)
    
    id = 0
    hyperparams = {'n_envs': N_ENVS}
    hyperparams['policy'] = 'MlpPolicy'
    hyperparams['policy_kwargs'] = architectures
    hyperparams['n_steps'] = N_STEPS
    hyperparams['gamma'] = GAMMA
    hyperparams['n_epochs'] = N_EPOCHS
    hyperparams['batch_size'] = BATCH_SIZE
    
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
        if ENABLE_GO_TO_REST:
            wrapper_list.append(go_to_rest_wrapper)
        if ENABLE_DEMO_WRAPPER:
            wrapper_list.append(demo_wrapper)
        wrapper_list.append(obs_flattening_wrapper)
        for env_config in possible_env_configs:
            for retrain_path in retrain_paths:
                kwargs = training_args.copy()
                kwargs['tensorboard_log'] = 'tensorlog'
                kwargs['env_kwargs'] = env_config
                # kwargs['curriculum'] = ENABLE_CURRICULUM
                if env_config['env_randomizer_mode'] == 'TEST_RANDOMIZER_CURRICULUM':
                    kwargs['curriculum'] = True
                else:
                    kwargs['curriculum'] = False
                hyperparam['env_wrapper'] = wrapper_list
                kwargs['hyperparams'] = hyperparam
                if RETRAIN_PREV_AGENT:
                    kwargs['trained_agent'] = retrain_path
                save_config(args=kwargs, id=id+last_id, path=PATH)
                id += 1
    
    print('***********************************')
    print(f'all possible env configs saved in:\n{PATH}')
    print('***********************************')
