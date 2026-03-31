"""Contrastive and matching losses used by the model."""

import torch
import torch.nn as nn
import json
import torch.nn.functional as F
from optim.loss.loss import LOSS_REGISTRY

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features

class ClipLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        if logit_bias is not None:
            logits_per_image += logit_bias
            logits_per_text += logit_bias

        return logits_per_image, logits_per_text

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            logit_bias=None,
            output_dict=False,
            bidirectional = True
    ):
        device = image_features.device

        logits_per_image, logits_per_text = self.get_logits(
            image_features,
            text_features,
            logit_scale,
            logit_bias=logit_bias,
        )

        labels = self.get_ground_truth(device, logits_per_image.shape[0])
        
        if bidirectional:
            total_loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
            ) / 2
        else:
            total_loss = F.cross_entropy(logits_per_image, labels)

        return {"contrastive_loss": total_loss} if output_dict else total_loss

@LOSS_REGISTRY.register()
class SceneViewPM_loss(nn.Module):
    def __init__(self, cfg, accelerator):
        super().__init__()
        self.accelerator = accelerator

        world_sz = self.accelerator.num_processes
        rank = self.accelerator.process_index
        self.contrast_loss = ClipLoss(rank=rank, world_size=world_sz)

        self.chamfer_ranking = load_json("dataset/chamfer_rankings.json")

        # ---- caches ----
        self._pos_idx_cache = {}         # (scan_id, V) -> LongTensor[V] on CPU
        self._dist_mat_cache = {}        # (scan_id, V) -> FloatTensor[V,V] on CPU
        self._view_keys_cache = {}       # scan_id -> sorted List[str]
        self._first_pos_cache = {}       # scan_id -> Dict[str, Optional[str]]
        self._eye_mask_cache = {}        # (V, device) -> BoolTensor[V,V]

        # ---- knobs for soft neighbor loss ----
        # tau_d: softness over neighbor ranks/distances (smaller=sharper)
        self.tau_d = float(getattr(cfg, "pm_nn_tau_d", 0.35)) if cfg is not None else 0.35
        # alpha: mix hard NN one-hot with soft distribution
        self.soft_alpha = float(getattr(cfg, "pm_nn_soft_alpha", 0.7)) if cfg is not None else 0.7

    # -------------------------
    # cached diag mask
    # -------------------------
    def _get_eye_mask(self, V: int, device: torch.device):
        key = (V, device)
        m = self._eye_mask_cache.get(key, None)
        if m is None:
            m = torch.eye(V, dtype=torch.bool, device=device)
            self._eye_mask_cache[key] = m
        return m

    # -------------------------
    # per-scan cache build
    # -------------------------
    def _ensure_scan_cached(self, scan_id: str):
        scan_id = str(scan_id)
        if scan_id in self._view_keys_cache and scan_id in self._first_pos_cache:
            return

        rank_dict = self.chamfer_ranking[scan_id]  # {view_key: [neighbors...]}

        # normalize keys to str
        keys = [str(k) for k in rank_dict.keys()]
        try:
            keys_sorted = sorted(keys, key=lambda x: int(x))
        except Exception:
            keys_sorted = keys

        # first non-self neighbor per key
        first_pos = {}
        for k in keys_sorted:
            ks = str(k)
            # tolerate int keys in json
            if ks in rank_dict:
                cands = rank_dict[ks]
            else:
                try:
                    cands = rank_dict[int(ks)]
                except Exception:
                    cands = []
            fp = None
            for v in cands:
                vs = str(v)
                if vs != ks:
                    fp = vs
                    break
            first_pos[ks] = fp

        self._view_keys_cache[scan_id] = keys_sorted
        self._first_pos_cache[scan_id] = first_pos

    # -------------------------
    # soft target helpers
    # -------------------------
    @staticmethod
    def _soft_targets_from_dist(dist_bvv: torch.Tensor, tau_d: float, eye: torch.Tensor, eps: float = 1e-8):
        """
        dist_bvv: (B,V,V) distances (>=0). Smaller = closer.
        eye: (V,V) bool diagonal mask
        returns: (B,V,V) row-stochastic soft targets, diag=0
        """
        # ensure diagonal excluded
        dist = dist_bvv.masked_fill(eye.unsqueeze(0), float("inf"))
        tgt_logits = -dist / max(tau_d, 1e-8)
        tgt_logits = tgt_logits.masked_fill(eye.unsqueeze(0), -1e9)

        p = torch.softmax(tgt_logits, dim=-1)  # (B,V,V)
        p = p.masked_fill(eye.unsqueeze(0), 0.0)
        p = p / (p.sum(dim=-1, keepdim=True) + eps)
        return p

    @staticmethod
    def _soft_ce_from_logits(logits_bvv: torch.Tensor, p_bvv: torch.Tensor):
        """Soft-label cross entropy: -sum p * log softmax(logits)"""
        log_q = F.log_softmax(logits_bvv, dim=-1)
        loss = -(p_bvv * log_q).sum(dim=-1)  # (B,V)
        return loss.mean()

    # -------------------------
    # hard NN targets (existing)
    # -------------------------
    @torch.no_grad()
    def _get_pos_idx_cpu(self, scan_id: str, V: int) -> torch.LongTensor:
        """
        CPU LongTensor [V], where pos_idx[i] is index of closest *other* view for view i.
        Cached per (scan_id, V).
        """
        scan_id = str(scan_id)
        key = (scan_id, V)
        cached = self._pos_idx_cache.get(key, None)
        if cached is not None:
            return cached

        self._ensure_scan_cached(scan_id)

        view_keys = self._view_keys_cache[scan_id][:V]
        key2pos = {k: i for i, k in enumerate(view_keys)}

        first_pos = self._first_pos_cache[scan_id]
        pos_idx = [0] * V

        for i, cur in enumerate(view_keys):
            cur = str(cur)
            pv = first_pos.get(cur, None)
            if pv is None or pv not in key2pos:
                pos_idx[i] = (i + 1) % V
            else:
                pos_idx[i] = key2pos[pv]

        out = torch.tensor(pos_idx, dtype=torch.long, device="cpu")
        self._pos_idx_cache[key] = out
        return out

    # -------------------------
    # NEW: distance matrix from ranking (soft neighbors)
    # -------------------------
    @torch.no_grad()
    def _get_rank_dist_cpu(self, scan_id: str, V: int) -> torch.FloatTensor:
        """
        Build a (V,V) "distance" matrix from chamfer ranking list:
          dist[i,j] = rank_position_of_view_j_in_view_i_neighbor_list
        Smaller = closer. Missing entries get large distance.
        Cached per (scan_id, V).
        """
        scan_id = str(scan_id)
        key = (scan_id, V)
        cached = self._dist_mat_cache.get(key, None)
        if cached is not None:
            return cached

        self._ensure_scan_cached(scan_id)
        rank_dict = self.chamfer_ranking[scan_id]

        view_keys = self._view_keys_cache[scan_id][:V]
        key2pos = {k: i for i, k in enumerate(view_keys)}

        # large distance for missing edges
        BIG = float(V + 50)

        dist = torch.full((V, V), BIG, dtype=torch.float32, device="cpu")

        for i, ki in enumerate(view_keys):
            kis = str(ki)
            # read neighbor ranking list
            if kis in rank_dict:
                neigh = rank_dict[kis]
            else:
                try:
                    neigh = rank_dict[int(kis)]
                except Exception:
                    neigh = []

            # map neighbor -> rank (skip self)
            # rank starts at 0 for closest neighbor
            r = 0
            for nb in neigh:
                nbs = str(nb)
                if nbs == kis:
                    continue
                j = key2pos.get(nbs, None)
                if j is None:
                    continue
                dist[i, j] = float(r)
                r += 1
                if r >= V:  # no need to go too deep
                    break

        # diagonal zero (won't be used; will be masked)
        dist.fill_diagonal_(0.0)

        self._dist_mat_cache[key] = dist
        return dist

    def forward(self, data_dict):
        scan_ids = list(data_dict["scan_id"])
        logit_scale = data_dict["logit_scale"]

        view_rgb = data_dict["inter_view_rgb_embed"]              # (B,V,D)
        view_txt = data_dict["inter_view_txt_embed"]              # (B,V,D)
        view_pm  = data_dict["inter_view_pm_embed"]               # (B,V,D)
        view_ground_txt = data_dict["inter_view_ground_txt_embed"]# (B,V,D)

        scene_rgb = data_dict["scene_rgb_embed"]                  # (B,D)
        scene_txt = data_dict["scene_text_embed"]                 # (B,D)
        scene_pm  = data_dict["scene_pm_embed"]                   # (B,D)

        B, V, D = view_pm.shape
        device = view_pm.device
        has_pointmap_input = data_dict.get(
            "has_pointmap_input",
            torch.ones(B, dtype=torch.bool, device=device),
        ).to(device=device, dtype=torch.bool)

        # ---- Normalize ----
        view_pm_norm         = F.normalize(view_pm,         p=2, dim=-1)   # (B,V,D)
        view_rgb_norm        = F.normalize(view_rgb,        p=2, dim=-1)
        view_txt_norm        = F.normalize(view_txt,        p=2, dim=-1)
        # view_context_pm_norm = F.normalize(view_context_pm, p=2, dim=-1)
        view_ground_txt_norm = F.normalize(view_ground_txt, p=2, dim=-1)

        scene_rgb = F.normalize(scene_rgb, p=2, dim=-1)
        scene_txt = F.normalize(scene_txt, p=2, dim=-1)
        scene_pm  = F.normalize(scene_pm,  p=2, dim=-1)

        # ---- Flatten for batch-wide CLIP losses (unchanged) ----
        # view_pm_context_f  = view_context_pm_norm.reshape(-1, D)   # (B*V,D)
        view_pm_f  = view_pm_norm.reshape(-1, D)   # (B*V,D)
        view_rgb_f = view_rgb_norm.reshape(-1, D)
        view_txt_f = view_txt_norm.reshape(-1, D)

        # ---- Cross-modal CLIP losses (batch-wide, unchanged) ----
        loss_rgb_pm_view  = self.contrast_loss(view_pm_f, view_rgb_f, logit_scale)
        loss_txt_pm_view  = self.contrast_loss(view_pm_f, view_txt_f, logit_scale)
        loss_rgb_pm_scene = self.contrast_loss(scene_pm,  scene_rgb,  logit_scale)
        loss_txt_pm_scene = self.contrast_loss(scene_pm,  scene_txt,  logit_scale)

        # ============================================================
        # Grounded contrastive (WITHIN-SCENE ONLY):
        # context PM (anchor) <-> grounded text, negatives are other views in the SAME scene
        # ============================================================
        gt_logits = torch.bmm(view_pm_norm, view_ground_txt_norm.transpose(1, 2)) * logit_scale  # (B,V,V)
        # gt_logits = torch.bmm(view_context_pm_norm, view_ground_txt_norm.transpose(1, 2)) * logit_scale  # (B,V,V)

        # mask for views that actually have grounded text (pass from dataloader ideally)
        ground_mask = data_dict.get(
            "inter_view_ground_txt_mask",
            torch.ones((B, V), dtype=torch.bool, device=device)
        )  # (B,V) bool

        targets_diag = torch.arange(V, device=device).unsqueeze(0).expand(B, V)  # (B,V)

        loss_ctxpm2gt = F.cross_entropy(
            gt_logits.reshape(B * V, V),
            targets_diag.reshape(B * V),
            reduction="none",
        ).reshape(B, V)

        loss_gt2ctxpm = F.cross_entropy(
            gt_logits.transpose(1, 2).reshape(B * V, V),
            targets_diag.reshape(B * V),
            reduction="none",
        ).reshape(B, V)

        den = ground_mask.float().sum().clamp_min(1.0)
        loss_grounded_view = (
            ((loss_ctxpm2gt + loss_gt2ctxpm) * 0.5) * ground_mask.float()
        ).sum() / den

        # ---- masks/constants ----
        eye = self._get_eye_mask(V, device)  # (V,V) bool
        NEG = -1e4 if view_pm_f.dtype in (torch.float16, torch.bfloat16) else -1e9

        # ============================================================
        # PM-PM geometric alignment loss
        # ============================================================
        logits_pm_pm = torch.bmm(view_pm_norm, view_pm_norm.transpose(1, 2)) * logit_scale  # (B,V,V)
        logits_pm_pm = logits_pm_pm.masked_fill(eye.unsqueeze(0), NEG)

        dist_cpu = [self._get_rank_dist_cpu(str(s), V) for s in scan_ids]  # list of (V,V) CPU
        dist_bvv = torch.stack(dist_cpu, dim=0).to(device, non_blocking=True)  # (B,V,V)

        p_soft = self._soft_targets_from_dist(dist_bvv, tau_d=self.tau_d, eye=eye)

        pos_idx_cpu = [self._get_pos_idx_cpu(str(s), V) for s in scan_ids]
        targets = torch.stack(pos_idx_cpu, dim=0).to(device, non_blocking=True)
        hard = F.one_hot(targets, num_classes=V).float()  # (B,V,V)
        hard = hard.masked_fill(eye.unsqueeze(0), 0.0)
        p_mix = self.soft_alpha * hard + (1.0 - self.soft_alpha) * p_soft
        p_mix = p_mix / (p_mix.sum(dim=-1, keepdim=True) + 1e-8)

        if has_pointmap_input.any():
            loss_pm_pm_view = self._soft_ce_from_logits(
                logits_pm_pm[has_pointmap_input],
                p_mix[has_pointmap_input],
            )
        else:
            loss_pm_pm_view = view_pm_norm.sum() * 0.0

        geo_w = 1.0
        loss_pm_pm_view = geo_w * loss_pm_pm_view
        
        return {
            "loss_rgb_pm_view":   loss_rgb_pm_view,
            "loss_txt_pm_view":   loss_txt_pm_view,
            "loss_rgb_pm_scene":  loss_rgb_pm_scene,
            "loss_txt_pm_scene":  loss_txt_pm_scene,
            "loss_geo_pm_nn":     loss_pm_pm_view,
            "loss_grounded_view": loss_grounded_view,
        }
