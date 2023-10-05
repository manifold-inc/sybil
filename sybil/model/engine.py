import torch
import torch.nn as nn

from typing import List
from imagebind.models import imagebind_model
from transformers import MistralForCausalLM, StoppingCriteria, StoppingCriteriaList
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
        
        # initalize the visual encoder ( imagebind )
        self.visual_hidden_size = 1024
        self.visual_encoder = imagebind_model.imagebind_huge(pretrained=True)

        # freeze the visual encoder
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder.eval()

        print('Visual encoder initialized.')

        # load the LLM
        self.llm = MistralForCausalLM.from_pretrained(pretrained_llm)

        if freeze_lm:
            for name, param in self.llm.named_parameters():
                param.requires_grad = False
            self.llm.eval()
        else:
            print("Instruct tuning the LLaMa ...")
            # add the lora module
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.args['lora_r'],
                lora_alpha=self.args['lora_alpha'],
                lora_dropout=self.args['lora_dropout'],
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
            )

            self.llm = get_peft_model(self.llm, peft_config)
            self.llm.print_trainable_parameters()
        print('Language decoder initialized.')


    def forward(self, batch):
        return batch