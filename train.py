import os

import sybil as sb


# check if the imagebind model is available at ./ckpts/imagebind/imagebind_huge.pth if not, download it from https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth

# imagebind_ckpt = sb.load_imagebind()

# config = {
#     'imagebind_ckpt': imagebind_ckpt,
# }


pretrained_llm = "teknium/CollectiveCognition-v1.1-Mistral-7B"
freeze_lm = True
freeze_input_proj = True
image_diffusion = "stabilityai/stable-diffusion-xl-base-1.0"
video_diffusion = "cerspense/zeroscope_v2_576w"
audio_diffusion = "cvssp/audioldm-m-full"

# # ========= text-to-image alignment tuning ========== #
n_img_tokens = 4
text_emb_to_img_layers = [-1]
num_gen_img_tokens = 4
text_fc_to_img_mode = "transformer"
# # ========= text-to-video alignment tuning ========== #
n_video_tokens = 24
text_emb_to_video_layers = [-1]
num_gen_video_tokens = 24
text_fc_to_video_mode = "transformer"

# # ========= text-to-audio alignment tuning ========== #
n_audio_tokens = 8
text_emb_to_audio_layers = [-1]
num_gen_audio_tokens = 8
text_fc_to_audio_mode = "transformer"

# # ========= other configs ========== #
seed = 13
max_length = 512  # max length of the user input prompt
logging_step = 5
num_clip_tokens = 77
gen_emb_dim = 768

config = {
    "pretrained_llm": pretrained_llm,
    "freeze_lm": freeze_lm,
    "freeze_input_proj": freeze_input_proj,

    "image_diffusion": image_diffusion,
    "video_diffusion": video_diffusion,
    "audio_diffusion": audio_diffusion,

    "n_img_tokens": n_img_tokens,
    "text_emb_to_img_layers": text_emb_to_img_layers,
    "num_gen_img_tokens": num_gen_img_tokens,
    "text_fc_to_img_mode": text_fc_to_img_mode,

    "n_video_tokens": n_video_tokens,
    "text_emb_to_video_layers": text_emb_to_video_layers,
    "num_gen_video_tokens": num_gen_video_tokens,
    "text_fc_to_video_mode": text_fc_to_video_mode,

    "n_audio_tokens": n_audio_tokens,
    "text_emb_to_audio_layers": text_emb_to_audio_layers,
    "num_gen_audio_tokens": num_gen_audio_tokens,
    "text_fc_to_audio_mode": text_fc_to_audio_mode,

    "seed": seed,
    "max_length": max_length,
    "logging_step": logging_step,
    "num_clip_tokens": num_clip_tokens,
    "gen_emb_dim": gen_emb_dim,
    
}
sybil = sb.engine(config)
