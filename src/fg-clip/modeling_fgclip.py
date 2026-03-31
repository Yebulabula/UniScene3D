import torch
import torch.nn as nn
import math

# from transformers import CLIPConfig,AutoConfig
from typing import Any, Optional, Tuple, Union
import torch.distributed.nn as nn_dist
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from typing import Tuple, Union
from .modeling_clip import CLIPModel, CLIPTextTransformer, CLIPVisionTransformer, CLIPOutput, CLIPAttention, CLIPMLP

import torch.distributed as dist
from torch.nn import AvgPool2d
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .modeling_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from torch import nn, einsum
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
import math
from torchvision.ops import roi_align


class FGCLIPConfig(CLIPConfig):
    model_type = "clip"

class FGCLIPModel(CLIPModel):
    config_class = FGCLIPConfig
    main_input_name = "text_long"

    def __init__(self, config):
        super(CLIPModel, self).__init__(config)

        if not isinstance(config.text_config, CLIPTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type CLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config
        text_config.eos_token_id = 49407
        text_config.pad_token_id = 49407
        text_config.bos_token_id = 49406

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = CLIPTextTransformer(text_config)

        self.vision_model = CLIPVisionTransformer(vision_config)
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)


        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.text_filip_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)


        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))
        self.logit_scale_finegraind = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))
        self.logit_scale_hardneg = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

 
        self.embed_dim = text_config.hidden_size
        self.world_size = 0

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        mask = None,
        use_patch_embedding: bool = True,
        use_geo_embedding: bool = True,
    ) -> torch.FloatTensor:

        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mask = mask,
            use_patch_embedding=use_patch_embedding,
            use_geo_embedding=use_geo_embedding,
        )
        
        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return vision_outputs.last_hidden_state, image_features
    
    def get_image_box_roi_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        box_info=None,
    ) -> torch.FloatTensor:


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict
        )

        bs = pixel_values.shape[0]
        length = vision_outputs[0].shape[1]-1
        h = int(math.sqrt(length))
        w = h

        feature_map = vision_outputs.hidden_states[-2]#[:, 1:, :]
        feature_map = self.forward_without_attn(feature_map)[:, 1:]

        feature_map = self.vision_model.post_layernorm(feature_map)
        feature_map = self.visual_projection(feature_map)

        feature_map = feature_map.view(bs, h, w, -1).permute(0, 3, 1, 2)
        x_rois = roi_align(feature_map.type(torch.float32),box_info, (1, 1), 1.0, -1, True)[..., 0, 0]


        return x_rois

    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        walk_short_pos: Optional[bool] = True,
        use_bbox: Optional[bool] = False
    ) -> torch.FloatTensor:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        pos_flag = walk_short_pos or use_bbox

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            walk_short_pos=pos_flag,
        )
        pooled_output = text_outputs[1]

        if walk_short_pos:
            text_features = self.text_projection(pooled_output)
        else:
            text_features = self.text_filip_projection(pooled_output)           

        return text_features

    @staticmethod
    def _denormalize_boxes(normed_boxes, x):
        h, w = x.shape[-2:]
        denormed_boxes = []
        for boxes in normed_boxes:

            new_boxes = boxes.clone()   # FIXME: do not change the value in normed_boxes!
            new_boxes[:, [0, 2]] *= w
            new_boxes[:, [1, 3]] *= h
            denormed_boxes.append(new_boxes.type(torch.float32))
        return denormed_boxes

    def forward_without_attn(self, x):
        # get last layer 
        residual = x
        x = self.vision_model.encoder.layers[-1].layer_norm1(x)

        x = F.linear(input=x, weight=self.vision_model.encoder.layers[-1].self_attn.v_proj.weight, bias=self.vision_model.encoder.layers[-1].self_attn.v_proj.bias)
        x = self.vision_model.encoder.layers[-1].self_attn.out_proj(x)
        x = residual+x

        residual = x
        x = self.vision_model.encoder.layers[-1].layer_norm2(x)
        x = self.vision_model.encoder.layers[-1].mlp(x)
        x = residual + x

        return x


    def get_image_dense_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding=False,
        box_info=None,
    ) -> torch.FloatTensor:

        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )


        bs = pixel_values.shape[0]
        length = vision_outputs[0].shape[1]-1
        h = int(math.sqrt(length))
        w = h

        feature_map = vision_outputs.hidden_states[-2]#[:, 1:, :]
        feature_map = self.forward_without_attn(feature_map)[:, 1:]

        feature_map = self.vision_model.post_layernorm(feature_map)
        feature_map = self.visual_projection(feature_map)

        return feature_map

