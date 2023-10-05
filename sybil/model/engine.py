import torch
import torch.nn as nn

from typing import List
from sybil.model.layers import TextFcLayer
from imagebind.models import imagebind_model
from transformers import MistralForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import LoraConfig, TaskType, get_peft_model

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops: List = None, encounters: int = 1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            _stop = torch.tensor(stop).to(input_ids[0].device)
            indices = torch.where(_stop[0] == input_ids)
            for i in indices:
                if len(i) > 0:
                    if torch.all(input_ids[0][i:i + len(_stop)] == _stop):
                        stop_count += 1
        if stop_count >= self.ENCOUNTERS:
            return True
        return False


class Engine(nn.Module):
    def __init__(self, config):
        super(Engine, self).__init__()
        self.config = config
        pretrained_llm = self.config["pretrained_llm"]
        freeze_lm = self.config["freeze_lm"]
        freeze_input_proj = self.config["freeze_input_proj"]
        
        # # initalize the visual encoder ( imagebind )
        # self.visual_hidden_size = 1024
        # self.visual_encoder = imagebind_model.imagebind_huge(pretrained=True)

        # # freeze the visual encoder
        # for name, param in self.visual_encoder.named_parameters():
        #     param.requires_grad = False
        # self.visual_encoder.eval()

        # print('Visual encoder initialized.')

        # # load the LLM
        # self.llm = MistralForCausalLM.from_pretrained(pretrained_llm)

        # if freeze_lm:
        #     for name, param in self.llm.named_parameters():
        #         param.requires_grad = False
        #     self.llm.eval()
        # else:
        #     print("Instruct tuning the LLaMa ...")
        #     # add the lora module
        #     peft_config = LoraConfig(
        #         task_type=TaskType.CAUSAL_LM,
        #         inference_mode=False,
        #         r=self.config['lora_r'],
        #         lora_alpha=self.config['lora_alpha'],
        #         lora_dropout=self.config['lora_dropout'],
        #         target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
        #     )

        #     self.llm = get_peft_model(self.llm, peft_config)
        #     self.llm.print_trainable_parameters()
        # print('Language decoder initialized.')


        # Set up the tokenizer
        self.llm_tokenizer = AutoTokenizer.from_pretrained(pretrained_llm)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        self.llm_tokenizer.padding_side = "right"



        # TODO: add the other modalities tokens


        # create projection layers
        self.llm_projection = nn.Linear(self.visual_hidden_size, self.llm.config.hidden_size)
        if freeze_input_proj:
            for name, param in self.llm_projection.named_parameters():
                param.requires_grad = False
            self.llm_projection.eval()
        
        self.input_embeddings = self.llm.get_input_embeddings()

        # TODO: add alignment modules for other modalities
        # the alignment module for LLM-TO-IMAGE
        self.sd_ckpt_path = self.config['image_diffusion']
        self.gen_text_hidden_fcs = nn.ModuleList([])
        for layer_idx in self.config['text_emb_to_img_layers']:
            if layer_idx == -1 or layer_idx == self.llm.config.num_hidden_layers:
                in_dim = self.llm.config.hidden_size

                self.gen_text_hidden_fcs.append(
                    TextFcLayer(in_dim, 768, num_input_tokens=self.config['num_gen_img_tokens'],
                                num_output_tokens=self.config['num_clip_tokens'],
                                mode=self.config['text_fc_to_img_mode']))
            # self.sd_pipe.text_encoder.config.hidden_size
            elif layer_idx < self.llm.config.num_hidden_layers:
                self.gen_text_hidden_fcs.append(
                    TextFcLayer(self.llm.config.hidden_size, 768,
                                num_input_tokens=self.config['num_gen_img_tokens'],
                                num_output_tokens=self.config['num_clip_tokens'],
                                mode=self.config['text_fc_to_img_mode']))
            else:
                raise ValueError(
                    f'Embedding of layer {layer_idx} was requested but model only has {self.llm.config.num_hidden_layers} layers.')

        # the alignment module for LLM-TO-VIDEO
        self.vd_ckpt_path = self.config['video_diffusion']
        self.gen_text_hidden_fcs_video = nn.ModuleList([])
        for layer_idx in self.config['text_emb_to_video_layers']:
            if layer_idx == -1 or layer_idx == self.llm.config.num_hidden_layers:
                in_dim = self.llm.config.hidden_size  # 4096

                self.gen_text_hidden_fcs_video.append(
                    TextFcLayer(in_dim, 1024, num_input_tokens=self.config['num_gen_video_tokens'],
                                num_output_tokens=self.config['num_clip_tokens'],
                                mode=self.config['text_fc_to_video_mode']))
            # self.vd_pipe.text_encoder.config.hidden_size
            elif layer_idx < self.llm.config.num_hidden_layers:
                self.gen_text_hidden_fcs_video.append(
                    TextFcLayer(self.llm.config.hidden_size, 1024,
                                num_input_tokens=self.config['num_gen_video_tokens'],
                                num_output_tokens=self.config['num_clip_tokens'],
                                mode=self.config['text_fc_to_video_mode']))
            else:
                raise ValueError(
                    f'Embedding of layer {layer_idx} was requested but model only has {self.llm.config.num_hidden_layers} layers.')

        # the alignment module for LLM-TO-AUDIO
        self.ad_ckpt_path = self.config['audio_diffusion']
        self.gen_text_hidden_fcs_audio = nn.ModuleList([])
        for layer_idx in self.config['text_emb_to_audio_layers']:
            if layer_idx == -1 or layer_idx == self.llm.config.num_hidden_layers:
                in_dim = self.llm.config.hidden_size

                self.gen_text_hidden_fcs_audio.append(
                    TextFcLayer(in_dim, 512,
                                num_input_tokens=self.config['num_gen_audio_tokens'],
                                num_output_tokens=1,
                                mode=self.config['text_fc_to_audio_mode']))
            # self.ad_pipe.text_encoder.config.projection_dim
            elif layer_idx < self.llm.config.num_hidden_layers:
                self.gen_text_hidden_fcs_audio.append(
                    TextFcLayer(self.llm.config.hidden_size, 512,
                                num_input_tokens=self.config['num_gen_audio_tokens'],
                                num_output_tokens=1,
                                mode=self.config['text_fc_to_audio_mode']))
            else:
                raise ValueError(
                    f'Embedding of layer {layer_idx} was requested but model only has {self.llm.config.num_hidden_layers} layers.')





    def forward(self, batch):
        return batch