import os

import time
import torch
import random
import pprint
import logging
import argparse
import deepspeed

import numpy as np
import sybil as sb

from transformers.deepspeed import HfDeepSpeedConfig


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', type=str, default='sybil')
    parser.add_argument('--mode', type=str, default='train', help='train or test or validation')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--save_path', type=str, default='./ckpt/delta_ckpt/sybil/sybil-7b-v0/')
    parser.add_argument('--log_path', type=str, default='./ckpt/delta_ckpt/sybil/sybil-7b-v0/log/')
    parser.add_argument('--assets_path', type=str, default='./assets/')

    # model configurations
    parser.add_argument('--max_length', type=int, default=512)  # the maximum input sequence length for LLMs
    parser.add_argument('--stage', type=int, default=1)  # the training stage
    parser.add_argument('--modality', type=list, default=['image', 'video', 'audio', 'text'])
    return parser.parse_args()


def initialize_distributed( args ):
    args['master_ip'] = os.getenv('MASTER_ADDR', 'localhost')
    args['master_port'] = os.getenv('MASTER_PORT', '6000')
    args['world_size'] = int(os.getenv('WORLD_SIZE', '1'))
    args['local_rank'] = int(os.getenv('RANK', '0')) % torch.cuda.device_count()
    device = args['local_rank'] % torch.cuda.device_count()
    torch.cuda.set_device(device)
    deepspeed.init_distributed(dist_backend='nccl')


def set_random_seed( seed ):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_environment(args):
    args['root_dir'] = './'
    # args['mode'] = 'train'
    config = sb.load_config(args)
    args.update(config)
    initialize_distributed(args)
    set_random_seed(args['seed'])


def build_directory( path ):
    if os.path.exists( path ):
        pass
    else:
        os.makedirs( path, exist_ok=True )


def run( **args ):
    # run create_environment
    create_environment( args ) 
    pprint.pprint( args )

    # set up deepspeed config path
    args['ds_config_path'] = f'./dsconfig/stage_{args["stage"]}.json'
    dschf = HfDeepSpeedConfig(args['ds_config_path'])
    args['dschf'] = dschf

    # create build directories
    build_directory(args['save_path'])
    build_directory(args['log_path'])

    if args['log_path']:
        logging.basicConfig(
            format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            level=logging.DEBUG,
            filename=f'{args["log_path"]}/train_{time.asctime()}.log',
            filemode='w'
        )
    train_data, train_iter, sampler = sb.load_dataset(args, args['dataset_name_list'])
    import code; code.interact(local=dict(globals(), **locals()))



args = parser_args()

run( **vars(args) ) 
# import code; code.interact(local=dict(globals(), **locals()))


# check if the imagebind model is available at ./ckpts/imagebind/imagebind_huge.pth if not, download it from https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth

# imagebind_ckpt = sb.load_imagebind()

# config = {
#     'imagebind_ckpt': imagebind_ckpt,
# }


# pretrained_llm = "teknium/CollectiveCognition-v1.1-Mistral-7B"
# freeze_lm = True
# freeze_input_proj = True
# image_diffusion = "stabilityai/stable-diffusion-xl-base-1.0"
# video_diffusion = "cerspense/zeroscope_v2_576w"
# audio_diffusion = "cvssp/audioldm-m-full"

# # # ========= text-to-image alignment tuning ========== #
# n_img_tokens = 4
# text_emb_to_img_layers = [-1]
# num_gen_img_tokens = 4
# text_fc_to_img_mode = "transformer"
# # # ========= text-to-video alignment tuning ========== #
# n_video_tokens = 24
# text_emb_to_video_layers = [-1]
# num_gen_video_tokens = 24
# text_fc_to_video_mode = "transformer"

# # # ========= text-to-audio alignment tuning ========== #
# n_audio_tokens = 8
# text_emb_to_audio_layers = [-1]
# num_gen_audio_tokens = 8
# text_fc_to_audio_mode = "transformer"

# # # ========= other configs ========== #
# seed = 13
# max_length = 512  # max length of the user input prompt
# logging_step = 5
# num_clip_tokens = 77
# gen_emb_dim = 768

# config = {
#     "pretrained_llm": pretrained_llm,
#     "freeze_lm": freeze_lm,
#     "freeze_input_proj": freeze_input_proj,

#     "image_diffusion": image_diffusion,
#     "video_diffusion": video_diffusion,
#     "audio_diffusion": audio_diffusion,

#     "n_img_tokens": n_img_tokens,
#     "text_emb_to_img_layers": text_emb_to_img_layers,
#     "num_gen_img_tokens": num_gen_img_tokens,
#     "text_fc_to_img_mode": text_fc_to_img_mode,

#     "n_video_tokens": n_video_tokens,
#     "text_emb_to_video_layers": text_emb_to_video_layers,
#     "num_gen_video_tokens": num_gen_video_tokens,
#     "text_fc_to_video_mode": text_fc_to_video_mode,

#     "n_audio_tokens": n_audio_tokens,
#     "text_emb_to_audio_layers": text_emb_to_audio_layers,
#     "num_gen_audio_tokens": num_gen_audio_tokens,
#     "text_fc_to_audio_mode": text_fc_to_audio_mode,

#     "seed": seed,
#     "max_length": max_length,
#     "logging_step": logging_step,
#     "num_clip_tokens": num_clip_tokens,
#     "gen_emb_dim": gen_emb_dim,

# }



# sybil = sb.engine(config)


