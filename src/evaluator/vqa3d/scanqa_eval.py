"""Evaluation logic for the ScanQA task."""

import os
import json
import collections
from pathlib import Path
import torch

from evaluator.common.build import EVALUATOR_REGISTRY, BaseEvaluator

from data.data_utils import ScanQAAnswer, clean_answer
from evaluator.common.build import EVALUATOR_REGISTRY


@EVALUATOR_REGISTRY.register()
class ScanQAEval(BaseEvaluator):
    """Evaluator for multiple-choice ScanQA predictions."""

    def __init__(self, cfg, accelerator, **kwargs):
        """Load the ScanQA answer vocabulary and metric settings."""
        self.target_metric = 'ans1_acc'
        self.save_dir = Path(cfg.exp_dir) / "eval_results" / self.__class__.__name__
        super().__init__(cfg, accelerator, **kwargs)

        self.color_initials = ["which color", "what color", "what colour", "which colour"]
        
        # if self.save:
        train_data = json.load(open(os.path.join(cfg.data.scan_family_base,
                                'annotations/scanqa/ScanQA_v1.0_train.json'), encoding='utf-8'))
        answer_counter = sum([data['answers'] for data in train_data], [])
        answer_counter = collections.Counter(sorted(answer_counter))
        answer_cands = answer_counter.keys()
        self.answer_vocab = ScanQAAnswer(answer_cands)

    def _contains_color(self, text: str) -> bool:
        """Check whether a question asks about color."""
        t = text.lower()
        return any(color in t for color in self.color_terms)  # self.color_terms is a list of color words/phrases

    def batch_metrics(self, data_dict, include_count=False):
        """Compute answer and object metrics for one batch."""
        metrics = {}
        total_count = len(data_dict['answer_scores'])
        # ans
        choice_1 = data_dict['answer_scores'].argmax(dim=-1)
        choice_10 = torch.topk(data_dict['answer_scores'].detach(), 10, -1)[1]
        
        correct1 = 0
        correct10 = 0
        non_color_correct1 = 0
        non_color_correct10 = 0
        non_color_total_count = 0
        
        for i in range(data_dict['answer_label'].shape[0]):
            question = data_dict['sentence'][i].lower()

            if not any(init in question for init in self.color_initials):
                non_color_total_count += 1
                
            if data_dict['answer_label'][i, choice_1[i]] == 1:
                if not any(init in question for init in self.color_initials):
                    non_color_correct1 += 1
                else:
                    print(question, data_dict['answer_label'][i].argmax().item(), choice_1[i])
        
                correct1 += 1
                
            for j in range(10):
                if data_dict['answer_label'][i, choice_10[i, j]] == 1:
                    correct10 += 1
                    if not any(init in question for init in self.color_initials):
                        non_color_correct10 += 1
                    break
                
        metrics['ans1_acc'] = correct1
        metrics['ans10_acc'] = correct10 
        metrics['non_color_ans1_acc'] = non_color_correct1
        metrics['non_color_ans10_acc'] = non_color_correct10 
        
        # get obj cls acc
        for key in data_dict.keys():
            if key.endswith('logits') and data_dict[key].ndim == 3 and data_dict[key].shape[:2] == data_dict['obj_labels'].shape:
                new_key = key.replace('logits', 'acc')
                pred = torch.argmax(data_dict[key], dim=2)
                gt = data_dict['obj_labels']
                mask = data_dict['obj_masks']
                metrics[new_key] = ((pred[mask] == gt[mask]).sum().item(), data_dict['obj_masks'].sum().item())

        for key in metrics:
            if isinstance(metrics[key], tuple):
                # already has count
                continue
            
            if 'non_color' in key:
                metrics[key] = (metrics[key], non_color_total_count)
            else:
                metrics[key] = (metrics[key], total_count)

        if self.save:
            for i in range(total_count):
                answer_top10 = [self.answer_vocab.itos(choice_10[i, j].item()) for j in range(10)]
                og3d_pred = torch.argmax(data_dict['og3d_logits'], dim=1)
                box = data_dict['obj_boxes'][i, og3d_pred[i]].cpu().numpy()
                pred_data = {
                    "scene_id": data_dict["scan_id"][i],
                    "question_id": data_dict["data_idx"][i],
                    "answer_top10": answer_top10
                }
                self.eval_results.append(pred_data)

        if not include_count:
            for key, v in metrics.items():
                metrics[key] = v[0] / max(v[1], 1)

        return metrics


@EVALUATOR_REGISTRY.register()
class ScanQAGenEval(ScanQAEval):
    """Evaluator for generative ScanQA predictions."""

    def __init__(self, cfg, accelerator, **kwargs):
        """Reuse the standard ScanQA evaluator setup."""
        super().__init__(cfg, accelerator, **kwargs)

    def batch_metrics(self, data_dict, include_count=False):
        """Score generated answers against the ground-truth list."""
        metrics = {}
        answer_preds = [clean_answer(a) for a in data_dict['answer_pred']]
        answer_gts = [list(map(clean_answer, a)) for a in data_dict['answers']]
        correct = len([1 for pred, gts in zip(answer_preds, answer_gts) if pred in gts])

        metrics['ans1_acc'] = (correct, len(answer_preds))

        if not include_count:
            for key, v in metrics.items():
                metrics[key] = v[0] / max(v[1], 1)
        
        return metrics
