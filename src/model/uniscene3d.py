"""Main UniScene3D model definition."""

import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast
from transformers import AutoModel, AutoTokenizer

from common.misc import build_fgclip_model_from_local_code_with_hf_weights
from model.build import MODEL_REGISTRY, BaseModel
from modules.build import build_module
from optim.utils import no_decay_param_group

@MODEL_REGISTRY.register()
class UniScene3D(BaseModel):
    """UniScene3D model for pretraining and downstream finetuning."""

    def __init__(self, cfg):
        """Build encoders and task heads for the selected mode."""
        super().__init__(cfg)
        self.cfg = cfg
        model_root = str(Path(__file__).resolve().parents[1] / "fg-clip")
        fgclip_repo_id = self.cfg.model.get("fgclip_repo_id", None)
        if fgclip_repo_id is None:
            self.pm_encoder = build_fgclip_model_from_local_code_with_hf_weights(model_root)
        else:
            self.pm_encoder = build_fgclip_model_from_local_code_with_hf_weights(model_root, repo_id=fgclip_repo_id)
        
        if cfg.mode == 'pretrain':
            if fgclip_repo_id is None:
                self.frozen_model = build_fgclip_model_from_local_code_with_hf_weights(model_root)
            else:
                self.frozen_model = build_fgclip_model_from_local_code_with_hf_weights(model_root, repo_id=fgclip_repo_id)
            self.use_scene_cap = self.cfg.data.args.get("use_scene_cap", False)
            # Pretraining uses a trainable encoder plus a frozen teacher copy.
            self.set_training_mode()
        else:
            self.text_encoder = AutoModel.from_pretrained('jinaai/jina-clip-v2', trust_remote_code=True)
            self.tokenizer =  AutoTokenizer.from_pretrained('jinaai/jina-clip-v2', trust_remote_code=True)
            self.text_encoder.text_model.output_tokens = True
            self.set_downstream_mode()

            self.head_list = self.cfg.model.heads.head_list
            for head in self.head_list:
                setattr(self, head, build_module("heads", getattr(self.cfg.model.heads, head)))

    def copy_patch_embed_to_geo_embed(self):
        """Initialize the point-map embed layer from the RGB patch embed."""
        with torch.no_grad():

            rgb_patch = self.pm_encoder.vision_model.embeddings.patch_embedding
            geo_patch = self.pm_encoder.vision_model.embeddings.geo_embedding

            # Copy weights
            geo_patch.weight.copy_(rgb_patch.weight)

            # Copy bias if exists
            if rgb_patch.bias is not None and geo_patch.bias is not None:
                geo_patch.bias.copy_(rgb_patch.bias)

        print("Geo embedding initialized from RGB patch embedding.")

    def freeze_module(self, module):
        """Disable gradients for every parameter in a module."""
        for param in module.parameters():
            param.requires_grad = False

    def set_pm_encoder_trainability(self, train_geo_embed_only=False):
        """Choose which FG-CLIP encoder weights stay trainable."""
        self.freeze_module(self.pm_encoder)

        for name, param in self.pm_encoder.named_parameters():
            if train_geo_embed_only:
                # This mode is useful when only the point-map patch embed should move.
                if "vision_model.embeddings.geo_embedding" in name:
                    param.requires_grad = True
            elif "text_model" not in name:
                param.requires_grad = True

    def set_training_mode(self):
        """Prepare the model for pretraining."""
        self.freeze_module(self.frozen_model)

        if not self.cfg.get("pretrain_ckpt_path"):
            self.copy_patch_embed_to_geo_embed()
        self.set_pm_encoder_trainability(train_geo_embed_only=False)

        self.pm_encoder.train()
        self.frozen_model.eval()

    def sync_geo_embedding_from_patch_after_load(self):
        """Optionally resync the geo embedding after checkpoint load."""
        if self.cfg.mode != "pretrain":
            return
        if self.cfg.model.get("copy_patch_embed_to_geo_after_load", False):
            self.copy_patch_embed_to_geo_embed()

    def apply_pretrain_modality_dropout(self, images, point_maps):
        """Randomly drop RGB or point-map inputs during pretraining."""
        dropout_cfg = self.cfg.model.get("modality_dropout", {})
        drop_rgb_prob = float(dropout_cfg.get("drop_rgb_prob", 0.0))
        drop_pm_prob = float(dropout_cfg.get("drop_pointmap_prob", 0.0))
        keep_both_prob = 1.0 - drop_rgb_prob - drop_pm_prob

        if drop_rgb_prob < 0.0 or drop_pm_prob < 0.0 or keep_both_prob < 0.0:
            raise ValueError(
                "modality_dropout probabilities must be >= 0 and sum to <= 1."
            )

        if (not self.training) or (drop_rgb_prob == 0.0 and drop_pm_prob == 0.0):
            return None

        batch_size = images.shape[0]
        modality_rand = torch.rand(batch_size, device=images.device)

        # Drop one modality per sample while keeping the other available.
        drop_rgb_mask = modality_rand < drop_rgb_prob
        drop_pm_mask = (modality_rand >= drop_rgb_prob) & (
            modality_rand < drop_rgb_prob + drop_pm_prob
        )

        dropout_stats = {
            "drop_rgb_count": int(drop_rgb_mask.sum().item()),
            "drop_pointmap_count": int(drop_pm_mask.sum().item()),
            "keep_both_count": int(batch_size - drop_rgb_mask.sum().item() - drop_pm_mask.sum().item()),
            "use_patch_embedding": ~drop_rgb_mask,
            "use_geo_embedding": ~drop_pm_mask,
        }
        return dropout_stats

    def augment_pretrain_pointmaps(self, point_maps):
        """Apply simple geometric augmentation to pretraining point maps."""
        aug_cfg = self.cfg.model.get("pointmap_augmentation", {})
        if (not self.training) or (not aug_cfg.get("enabled", False)):
            return point_maps

        rotations_deg = aug_cfg.get("z_rotations_deg", [90, 180, 270])
        if len(rotations_deg) == 0:
            return point_maps

        scale_min, scale_max = aug_cfg.get("scale_range", [0.75, 1.25])
        translate_min, translate_max = aug_cfg.get("translation_range", [-1.0, 1.0])

        batch_size, num_views, height, width, channels = point_maps.shape
        if channels != 3:
            raise ValueError(f"Expected point maps with 3 channels, got {channels}.")
        points = point_maps.float().reshape(batch_size, -1, channels)

        valid_mask = torch.isfinite(points).all(dim=-1) & (points.abs().sum(dim=-1) > 0)
        valid_points = torch.where(valid_mask.unsqueeze(-1), points, torch.zeros_like(points))
        valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp_min(1)
        # Rotate and scale around each sample centroid so geometry stays centered.
        centroids = valid_points.sum(dim=1) / valid_counts.float()

        rotation_choices = torch.tensor(
            rotations_deg,
            device=points.device,
            dtype=points.dtype,
        )
        rotation_indices = torch.randint(
            low=0,
            high=rotation_choices.numel(),
            size=(batch_size,),
            device=points.device,
        )
        angles = rotation_choices[rotation_indices] * (math.pi / 180.0)
        cos_theta = torch.cos(angles)
        sin_theta = torch.sin(angles)

        rotation_mats = torch.zeros(batch_size, 3, 3, device=points.device, dtype=points.dtype)
        rotation_mats[:, 0, 0] = cos_theta
        rotation_mats[:, 0, 1] = -sin_theta
        rotation_mats[:, 1, 0] = sin_theta
        rotation_mats[:, 1, 1] = cos_theta
        rotation_mats[:, 2, 2] = 1.0

        scales = torch.empty(batch_size, 1, 1, device=points.device, dtype=points.dtype).uniform_(
            float(scale_min), float(scale_max)
        )
        translations = torch.empty(batch_size, 1, 3, device=points.device, dtype=points.dtype).uniform_(
            float(translate_min), float(translate_max)
        )

        centered = points - centroids.unsqueeze(1)
        rotated = torch.einsum("bnc,bdc->bnd", centered, rotation_mats)
        augmented_points = rotated * scales + centroids.unsqueeze(1) + translations
        points = torch.where(valid_mask.unsqueeze(-1), augmented_points, points)

        return points.reshape(batch_size, num_views, height, width, channels).to(point_maps.dtype)

    def set_downstream_mode(self):
        """Freeze vision weights and enable downstream text training."""
        for param in self.pm_encoder.parameters():
            param.requires_grad = False
            
        for name, param in self.text_encoder.named_parameters():
            if "vision_model" in name:
                param.requires_grad = False
                
        self.pm_encoder.eval()
        self.text_encoder.train()
        
    def forward(self, data_dict, mode=None):        
        """Run the pretraining or QA forward pass."""
        if 'cur_step' not in data_dict:
            data_dict['cur_step'] = 1
            data_dict['total_steps'] = 1
    
        data_dict['logit_scale'] = self.pm_encoder.logit_scale.exp()

        if mode == 'pretrain':
            pm_basic_features = []
            B, V, H, W, C = data_dict['point_map'].shape
            
            data_dict['point_map'] = self.augment_pretrain_pointmaps(
                data_dict['point_map'].to(torch.float32, non_blocking=True)
            )
            data_dict['point_map'] = data_dict['point_map'].to(torch.bfloat16, non_blocking=True).permute(0, 1, 4, 2, 3)
            dropout_stats = self.apply_pretrain_modality_dropout(
                data_dict['images'],
                data_dict['point_map'],
            )
            if dropout_stats is not None:
                data_dict["drop_rgb_count"] = dropout_stats["drop_rgb_count"]
                data_dict["drop_pointmap_count"] = dropout_stats["drop_pointmap_count"]
                data_dict["keep_both_count"] = dropout_stats["keep_both_count"]
                data_dict["has_pointmap_input"] = dropout_stats["use_geo_embedding"]
            else:
                data_dict["has_pointmap_input"] = torch.ones(B, dtype=torch.bool, device=data_dict["point_map"].device)
            
            # Encode each scene independently because the underlying model is view-based.
            for i in range(data_dict['point_map'].shape[0]): 
                with autocast("cuda", dtype=torch.bfloat16):
                    pm = data_dict['point_map'][i] 
                    images = data_dict['images'][i]
                    use_patch_embedding = True
                    use_geo_embedding = True
                    if dropout_stats is not None:
                        use_patch_embedding = bool(dropout_stats["use_patch_embedding"][i].item())
                        use_geo_embedding = bool(dropout_stats["use_geo_embedding"][i].item())
                        
                    color_pm = torch.cat([images, pm], dim=1)
                    _, pm_feat = self.pm_encoder.get_image_features(
                        color_pm,
                        use_patch_embedding=use_patch_embedding,
                        use_geo_embedding=use_geo_embedding,
                    )
                    pm_basic_features.append(pm_feat)
                
            pm_basic_features = torch.stack(pm_basic_features, dim=0) 
            data_dict['inter_view_pm_embed'] = pm_basic_features
            
            data_dict['scene_pm_embed'] = data_dict['inter_view_pm_embed'].mean(dim=1)
            
            B_txt = data_dict['txt_ids'].shape[0]
            lang_basic_features = torch.empty((B_txt, 32, 512), dtype=torch.bfloat16, device=data_dict['txt_ids'].device)
            ground_lang_basic_features = torch.empty((B_txt, 32, 512), dtype=torch.bfloat16, device=data_dict['txt_ids'].device)
            rgb_basic_features  = torch.empty((B_txt, 32, 512), dtype=torch.bfloat16, device=data_dict['txt_ids'].device)
            with torch.no_grad():
                with autocast("cuda", dtype=torch.bfloat16):
                    for i in range(B_txt):
                        lang_basic_features[i] = self.frozen_model.get_text_features(data_dict['txt_ids'][i], walk_short_pos=True)
                        ground_lang_basic_features[i] = self.frozen_model.get_text_features(data_dict['ground_txt_ids'][i], walk_short_pos=True)
                        rgb_basic_features[i]  = self.frozen_model.get_image_features(data_dict['images'][i])[1]

                    if getattr(self, "use_scene_cap", False):
                        data_dict['scene_text_embed'] = self.frozen_model.get_text_features(data_dict['scene_txt_ids'], walk_short_pos=False)
                    
            data_dict['inter_view_txt_embed'] = lang_basic_features
            data_dict['inter_view_ground_txt_embed'] = ground_lang_basic_features
            data_dict['inter_view_rgb_embed'] = rgb_basic_features
            data_dict['scene_rgb_embed'] = rgb_basic_features.mean(dim=1)
        elif mode == 'qa':
            B, V, C, H, W = data_dict['images'].shape
            images = data_dict['images'].reshape(B * V, C, H, W).contiguous().float()
            pm = data_dict['point_map'].reshape(B * V, C, H, W).contiguous().float()

            color_pm = torch.cat([images, pm], dim=1)
            with torch.no_grad():
                with autocast("cuda", dtype=torch.bfloat16):
                    _, vision_feat = self.pm_encoder.get_image_features(color_pm)
                    inter_view_pm_embed = vision_feat.reshape(B, V, -1)
                    
            tokenized = self.tokenizer.batch_encode_plus(
                data_dict['sentence'],
                padding="max_length",
                return_tensors="pt",
                max_length=256,
            ).to(inter_view_pm_embed.device)
            
            txt_ids = tokenized['input_ids']
            with autocast("cuda", dtype=torch.bfloat16):
                inter_view_txt_tokens = self.text_encoder.text_model(txt_ids)[-1]
                attention_mask = tokenized['attention_mask'].ne(1).bool()

                if hasattr(self, "qa_head") and self.qa_head is not None:
                    answer_scores = self.qa_head(
                        inter_view_pm_embed,
                        inter_view_txt_tokens,
                        attention_mask,
                    )
                    data_dict['answer_scores'] = answer_scores  
        return data_dict

    def get_text_params(self, model):
        """Return the text-encoder parameters."""
        text_params = [
            (n, p) for n, p in model.named_parameters()
            if "text_model" in n
        ]
        return text_params

    def get_pretrain_params(self):
        """Return trainable pretraining parameters from the vision encoder."""
        pretrain_params = [
            (n, p) for n, p in self.pm_encoder.named_parameters()
            if "text_model" not in n
        ]
        return pretrain_params

    def get_opt_params(self):
        """Build optimizer parameter groups for the current mode."""
        def get_lr(cfg, default_lr):
            return default_lr if cfg.get("lr") is None else cfg.get("lr")

        optimizer_grouped_parameters = []
        if self.cfg.mode == 'pretrain':
            optimizer_grouped_parameters += no_decay_param_group(
                self.get_pretrain_params(),
                get_lr(self.cfg.model.vision, self.cfg.solver.lr),
            )
        else:
            optimizer_grouped_parameters += no_decay_param_group(self.get_text_params(self.text_encoder), get_lr(self.cfg.model.vision, self.cfg.solver.lr))
            if "qa_head" in self.head_list:
                optimizer_grouped_parameters += no_decay_param_group(
                    self.qa_head.named_parameters(), get_lr(self.cfg.model.heads.qa_head, self.cfg.solver.lr)
            )
        return optimizer_grouped_parameters
