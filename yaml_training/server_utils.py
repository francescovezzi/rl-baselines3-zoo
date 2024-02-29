import glob
import yaml
import os, inspect, sys
from collections import OrderedDict


current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)

sys.path.insert(0, parent_dir)


def remove_after_2nd_last_char(name, char):
    splitted_name = name.split(char)
    if not splitted_name:
        raise ValueError(f'no {char} character found')
    elif len(splitted_name) > 2:
        splitted_name = splitted_name[:-1]
        splitted_name = char.join(splitted_name)
    else:
        splitted_name = char.join(splitted_name)
    return splitted_name

def get_id(name, env_id):
    base_name = name.split('/')[-1]
    splitted_name = base_name.split('_')
    ext = splitted_name[-1]
    if env_id == "_".join(splitted_name[-2:-1]) and ext.isdigit():
        id = int(ext)
    else:
        raise ValueError('no id found')
    return id

def save_config(args, id, path='server_scripts/train_env_kwargs'):
    with open(os.path.join(path, f"args_{id+1}.yml"), "w") as f:
        ordered_args = OrderedDict([(key, args[key]) for key in sorted(args.keys())])
        yaml.dump(ordered_args, f)

def get_args_id(name):
    _, last = name.split('_')
    id, _ = last.split('.')
    return int(id)

def get_last_args_id(path = 'server_scripts/train_env_kwargs'):
    max_id = 1;
    for file in glob.glob(os.path.join(path, 'args_*.yml')):
        file = file.split('/')[-1]
        max_id = max(max_id, get_args_id(file))
    return max_id

def gen_list_string(l):
    ret = []
    for el in l:
        new_el = el.split('"')[1]
        ret.append(new_el)
    return ret

def gen_dict_string(d):
    command = ''
    for k, v in d.items():
        if isinstance(v, list):
            v = gen_list_string(v)
            v = "".join(f'{v}'.split(' '))
        add = f'{k}:{v} '
        command += add
    return command

def gen_command(d):
    command = ''
    for k, v in d.items():
        if k == '--curriculum':
            value = ''
        else:
            if isinstance(v, dict):
                value = gen_dict_string(v)
            else:
                value = f'{v}'
        add = f'{k} {value} '
        command += add
    return command

def print_dict(d, name='configuration'):
    print('**********************************')
    print(f'{name}')
    for k, v in d.items():
        print(f'{k}: {v}')
    print('**********************************')

def adapt_args_for_command(args):
    new_args = {}
    new_args['--algo'] = args['algo']
    new_args['--env'] = args['env']
    new_args['--env-kwargs'] = adapt_string_kwargs(args['env_kwargs'])
    new_args['--eval-episodes'] = args['eval_episodes']
    new_args['--eval-freq'] = args['eval_freq']
    new_args['--log-interval'] = args['log_interval']
    new_args['--n-eval-envs'] = args['n_eval_envs']
    new_args['--save-freq'] = args['save_freq']
    new_args['-f'] = args['log_folder']
    new_args['-params'] = args['hyperparams']
    
    return new_args

def adapt_string_kwargs(args):
    for k, v in args.items():
        if isinstance(v, str):
            args[k] = f'"{v}"'
    # print(args)
    return args

if __name__ ==  "__main__":
    name = 'ofvn/sdbv/sdhvc/env-id_5'
    id = get_id(name, 'env-id')
    diz = {'s':2, 'w':8}
    print_dict(diz)
    # print(id)
