import os
import json
import argparse
import datetime
import numpy as np
from shutil import copyfile
import torch
import torch.multiprocessing as mp

from core.trainer import Trainer
from core.dist import (
    get_world_size,
    get_local_rank,
    get_global_rank,
    get_master_ip,
)

parser = argparse.ArgumentParser(description='STTN')
parser.add_argument('-c', '--config', default='configs/music.json', type=str)
parser.add_argument('-m', '--model', default='sttn', type=str)
parser.add_argument('-p', '--port', default='29500', type=str)
parser.add_argument('-e', '--exam', action='store_true')
args = parser.parse_args()


def main_worker(rank, config):
    if 'local_rank' not in config:
        config['local_rank'] = config['global_rank'] = rank

    config['save_dir'] = os.path.join(config['save_dir'], '{}_{}'.format(config['model'],
                                                                         os.path.basename(args.config).split('.')[0]))
    if torch.cuda.is_available(): 
        config['device'] = torch.device("cuda:{}".format(config['local_rank']))
    else: 
        config['device'] = 'cpu'

    if (not config['distributed']) or config['global_rank'] == 0:
        os.makedirs(config['save_dir'], exist_ok=True)
        config_path = os.path.join(
            config['save_dir'], config['config'].split('/')[-1])
        if not os.path.isfile(config_path):
            copyfile(config['config'], config_path)
        print('[**] create folder {}'.format(config['save_dir']))
    
    trainer = Trainer(config, debug=args.exam)
    trainer.train()


if __name__ == "__main__":    
    # loading configs
    config = json.load(open(args.config))
    config['model'] = args.model
    config['config'] = args.config

    # setting distributed configurations
    config['world_size'] = get_world_size()
    config['init_method'] = f"tcp://{get_master_ip()}:{args.port}"
    config['distributed'] = True if config['world_size'] > 1 else False

    main_worker(0, config)