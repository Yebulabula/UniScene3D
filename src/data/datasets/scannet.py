"""ScanNet datasets for pretraining and VQA tasks."""

# --- Standard library ---
import os
import json
import random
import collections

# --- Torch ---
import torch

# --- Project-specific ---
from ..build import DATASET_REGISTRY
from ..data_utils import (
    ScanQAAnswer,
    SQA3DAnswer,
    Hypo3DAnswer,
    msnnAnswer,
    get_sqa_question_type,
    load_safetensor_from_hf,
)
from .base import ScanBase
from .scannet_base import ScanNetBase


@DATASET_REGISTRY.register()
class ScanNetSpatialRefer(ScanBase):
    """ScanNet spatial-refer dataset used for pretraining."""

    def __init__(self, cfg, split):
        """Load ScanNet pretraining language data and scan metadata."""
        super(ScanNetSpatialRefer, self).__init__(cfg, split)
        self.base_dir = cfg.data.scan_family_base
        
        split_cfg = cfg.data.get(self.__class__.__name__).get(split)    
        all_scan_ids = self._load_split(self.split)
        self.lang_data, self.ground_lang_data, self.scan_ids = self._load_lang(split_cfg, all_scan_ids)
        if self.cfg.mode != 'pretrain':
            raise RuntimeError('ScanNetSpatialRefer downstream mode was removed because it is unused in this repo.')

        print(f"Loading ScanNet {split}-set scans")
        self.scan_data = self._load_scan_pretrain(self.lang_data, self.ground_lang_data)
        print(f"Finish loading ScanNet {split}-set scans")

    def __len__(self):
        """Return the number of scene-level samples."""
        return len(self.scan_data)

    def __getitem__(self, index):
        """Return one pretraining sample."""
        return self._getitem_refer(index)

@DATASET_REGISTRY.register()
class ScanNetHypo3D(ScanNetBase):
    """ScanNet dataset for the Hypo3D task."""

    def __init__(self, cfg, split):
        """Load Hypo3D annotations, questions, and scan metadata."""
        super().__init__(cfg, split)

        self.use_unanswer = cfg.data.get(self.__class__.__name__).get(split).use_unanswer

        assert cfg.data.args.sem_type in ['607']
        assert self.split in ['train', 'val', 'test']
        if self.split == 'val':
            self.split = 'test'

        print(f"Loading ScanNet Hypo3D {split}-set language")
        
        # build answer
        self.num_answers, self.answer_vocab, self.answer_cands = self.build_answer()

        # load annotations
        lang_data, self.scan_ids = self._load_lang()
        if cfg.debug.flag:
            self.lang_data = []
            self.scan_ids = sorted(list(self.scan_ids))[:cfg.debug.debug_size]
            for item in lang_data:
                if item['scene_id'] in self.scan_ids:
                    self.lang_data.append(item)
        else:
            self.lang_data = lang_data

        # load question engine
        self.questions_map = self._load_question()
        print(f"Finish loading ScanNet Hypo3D {split}-set language")

        # load scans
        print(f"Loading ScanNet Hypo3D {split}-set scans")
        self.scan_data = self._load_scannet(self.scan_ids)
        
        print(f"Finish loading ScanNet Hypo3D {split}-set data")

    def __getitem__(self, index):
        """Return one Hypo3D sample."""
        item = self.lang_data[index]
        item_id = item['question_id']
        scan_id = item['scene_id']

        answer_list = [answer['answer'] for answer in item['answers']]
        answer_id_list = [self.answer_vocab.stoi(answer)
                          for answer in answer_list if self.answer_vocab.stoi(answer) >= 0]

        if self.split == 'train':
            # augment with random situation for train
            context_change = random.choice(self.questions_map[scan_id][item_id]['context_change'])
        else:
            # fix for eval
            context_change = self.questions_map[scan_id][item_id]['context_change'][0]

        question = self.questions_map[scan_id][item_id]['question']

        orientation = self.questions_map[scan_id][item_id]['orientation']
        concat_sentence = orientation + context_change + question
        scene_tensor = load_safetensor_from_hf(
            repo_id="MatchLab/ScenePoint",
            filename=self.scan_data[scan_id]["safetensors_path"],
        )
        point_map = scene_tensor['point_map'].permute(0, 3, 1, 2)
        images = scene_tensor['color_images'].permute(0, 3, 1, 2)

        # convert answer format
        answer_label = torch.zeros(self.num_answers).long()
        for _id in answer_id_list:
            answer_label[_id] = 1
            
        data_dict = {
            "context_change": context_change,
            "question": question,
            "sentence": concat_sentence,
            "scan_dir": os.path.join(self.base_dir, 'scans'),
            "scan_id": scan_id,
            "answer": "[answer_seq]".join(answer_list),
            "answer_label": answer_label,
            "point_map": point_map,
            "images": images,
            "data_idx": item_id,
        }

        return data_dict

    def build_answer(self):
        """Build the Hypo3D answer vocabulary."""
        answer_data = json.load(
            open(os.path.join(self.base_dir,
                              'annotations/hypo3d/answer_dict.json'), encoding='utf-8')
            )[0]
        answer_counter = []
        for data in answer_data.keys():
            answer_counter.append(data)
        answer_counter = collections.Counter(sorted(answer_counter))
        num_answers = len(answer_counter)
        answer_cands = answer_counter.keys()
        answer_vocab = Hypo3DAnswer(answer_cands)
        print(f"total answers is {num_answers}")
        return num_answers, answer_vocab, answer_cands

    def _load_lang(self):
        """Load Hypo3D annotations for the current split."""
        lang_data = []
        scan_ids = set()
        anno_file = os.path.join(self.base_dir,
            f'annotations/hypo3d/balanced/hypo3d_{self.split}_annotations.json')
        json_data = json.load(open(anno_file, 'r', encoding='utf-8'))['annotations']
        for item in json_data:
            if self.use_unanswer or (len(set(item['answers']) & set(self.answer_cands)) > 0):
                scan_ids.add(item['scene_id'])
                lang_data.append(item)
        print(f'{self.split} unanswerable question {len(json_data) - len(lang_data)},'
              + f'answerable question {len(lang_data)}')

        return lang_data, scan_ids

    def _load_question(self):
        """Load Hypo3D question metadata keyed by scan and question id."""
        questions_map = {}
        anno_file = os.path.join(self.base_dir,
            f'annotations/hypo3d/balanced/hypo3d_{self.split}.json')
        json_data = json.load(open(anno_file, 'r', encoding='utf-8'))['questions']
        for item in json_data:
            if item['scene_id'] not in questions_map.keys():
                questions_map[item['scene_id']] = {}
            questions_map[item['scene_id']][item['question_id']] = {
                'context_change': [item['context_change']],   # list of sentences
                'orientation': item['orientation'],
                'question': item['question']
            }

        return questions_map


@DATASET_REGISTRY.register()
class ScanNetmsnn(ScanNetBase):
    """ScanNet dataset for the MSNN task."""

    def __init__(self, cfg, split):
        """Load MSNN annotations, questions, and scan metadata."""
        super().__init__(cfg, split)

        self.use_unanswer = cfg.data.get(self.__class__.__name__).get(split).use_unanswer

        assert cfg.data.args.sem_type in ['607']
        assert self.split in ['train', 'val', 'test']
        if self.split == 'val':
            self.split = 'test'

        print(f"Loading ScanNet MSNN {split}-set language")
        
        # build answer
        self.num_answers, self.answer_vocab, self.answer_cands = self.build_answer()

        # load annotations
        lang_data, self.scan_ids = self._load_lang()
        if cfg.debug.flag:
            self.lang_data = []
            self.scan_ids = sorted(list(self.scan_ids))[:cfg.debug.debug_size]
            for item in lang_data:
                if item['scene_id'] in self.scan_ids:
                    self.lang_data.append(item)
        else:
            self.lang_data = lang_data

        # load question engine
        self.questions_map = self._load_question()
        print(f"Finish loading ScanNet MSNN {split}-set language")

        # load scans
        print(f"Loading ScanNet MSNN {split}-set scans")
        self.scan_data = self._load_scannet(self.scan_ids)
        
        print(f"Finish loading ScanNet MSNN {split}-set data")

    def __getitem__(self, index):
        """Return one MSNN sample."""
        item = self.lang_data[index]
        item_id = item['question_id']
        scan_id = item['scan_id']

        answer_list = [answer['answer'] for answer in item['answers']]
        answer_id_list = [self.answer_vocab.stoi(answer)
                          for answer in answer_list if self.answer_vocab.stoi(answer) >= 0]

        situation = self.questions_map[scan_id][item_id]['situation_text']
        question = self.questions_map[scan_id][item_id]['question']
        interaction = self.questions_map[scan_id][item_id]['interaction']

        concat_sentence = situation + interaction + question
        scene_tensor = load_safetensor_from_hf(
            repo_id="MatchLab/ScenePoint",
            filename=self.scan_data[scan_id]["safetensors_path"],
        )
        point_map = scene_tensor['point_map'].permute(0, 3, 1, 2)
        images = scene_tensor['color_images'].permute(0, 3, 1, 2)

        # convert answer format
        answer_label = torch.zeros(self.num_answers).long()
        for _id in answer_id_list:
            answer_label[_id] = 1
            
        data_dict = {
            "situation": situation,
            "interaction": interaction,
            "question": question,
            "sentence": concat_sentence,
            "scan_dir": os.path.join(self.base_dir, 'scans'),
            "scan_id": scan_id,
            "answer": "[answer_seq]".join(answer_list),
            "answer_label": answer_label,
            "point_map": point_map,
            "images": images,
            "data_idx": item_id,
        }

        return data_dict

    def build_answer(self):
        """Build the MSNN answer vocabulary."""
        answer_path = os.path.join(self.base_dir, 'annotations/msnn/answer_dict.json')
        if os.path.exists(answer_path):
            answer_data = json.load(open(answer_path, encoding='utf-8'))[0]
            answer_counter = collections.Counter(sorted(answer_data.keys()))
        else:
            train_anno_path = os.path.join(
                self.base_dir,
                'annotations/msnn/balanced/msnn_train_four_direction_annotations.json',
            )
            train_data = json.load(open(train_anno_path, 'r', encoding='utf-8'))['annotations']
            all_answers = [answer['answer'] for item in train_data for answer in item['answers']]
            answer_counter = collections.Counter(sorted(all_answers))
        num_answers = len(answer_counter)
        answer_cands = answer_counter.keys()
        answer_vocab = msnnAnswer(answer_cands)
        print(f"total answers is {num_answers}")
        return num_answers, answer_vocab, answer_cands

    def _load_lang(self):
        """Load MSNN annotations for the current split."""
        lang_data = []
        scan_ids = set()
        anno_file = os.path.join(self.base_dir,
            f'annotations/msnn/balanced/msnn_{self.split}_four_direction_annotations.json')
        json_data = json.load(open(anno_file, 'r', encoding='utf-8'))['annotations']
        for item in json_data:
            if self.use_unanswer or (len(set(item['answers']) & set(self.answer_cands)) > 0):
                scan_ids.add(item['scan_id'])
                lang_data.append(item)
        print(f'{self.split} unanswerable question {len(json_data) - len(lang_data)},'
              + f'answerable question {len(lang_data)}')

        return lang_data, scan_ids

    def _load_question(self):
        """Load MSNN question metadata keyed by scan and question id."""
        questions_map = {}
        anno_file = os.path.join(self.base_dir,
            f'annotations/msnn/balanced/msnn_{self.split}_four_direction.json')
        json_data = json.load(open(anno_file, 'r', encoding='utf-8'))
        for item in json_data:
            if item['scan_id'] not in questions_map.keys():
                questions_map[item['scan_id']] = {}
            questions_map[item['scan_id']][item['question_id']] = {
                'situation_text': item['situation_text'],   # list of sentences
                'interaction': item['interaction'],
                'question': item['question']
            }

        return questions_map

@DATASET_REGISTRY.register()
class ScanNetSQA3D(ScanNetBase):
    """ScanNet dataset for the SQA3D task."""

    def __init__(self, cfg, split):
        """Load SQA3D annotations, questions, and scan metadata."""
        super().__init__(cfg, split)

        self.use_unanswer = cfg.data.get(self.__class__.__name__).get(split).use_unanswer

        assert self.split in ['train', 'val', 'test']
        if self.split == 'val':
            self.split = 'test'

        print(f"Loading ScanNet SQA3D {split}-set language")
        # build answer
        self.num_answers, self.answer_vocab, self.answer_cands = self.build_answer()
        
        # load annotations
        lang_data, self.scan_ids = self._load_lang()
        if cfg.debug.flag:
            self.lang_data = []
            self.scan_ids = sorted(list(self.scan_ids))[:cfg.debug.debug_size]
            for item in lang_data:
                if item['scene_id'] in self.scan_ids:
                    self.lang_data.append(item)
        else:
            self.lang_data = lang_data

        # load question engine
        self.questions_map = self._load_question()
        print(f"Finish loading ScanNet SQA3D {split}-set language")

        # load scans
        print(f"Loading ScanNet SQA3D {split}-set scans")

        self.scan_data = self._load_scannet(self.scan_ids)
        print(f"Finish loading ScanNet SQA3D {split}-set data")

    def __getitem__(self, index):
        """Return one SQA3D sample."""
        item = self.lang_data[index]
        item_id = item['question_id']
        scan_id = item['scene_id']

        answer_list = [answer['answer'] for answer in item['answers']]
        answer_id_list = [self.answer_vocab.stoi(answer)
                          for answer in answer_list if self.answer_vocab.stoi(answer) >= 0]

        if self.split == 'train':
            # augment with random situation for train
            situation = random.choice(self.questions_map[scan_id][item_id]['situation'])
        else:
            # fix for eval
            situation = self.questions_map[scan_id][item_id]['situation'][0]

        question = self.questions_map[scan_id][item_id]['question']
        concat_sentence = situation + question
        scene_tensor = load_safetensor_from_hf(repo_id="MatchLab/ScenePoint",filename=self.scan_data[scan_id]["safetensors_path"])
        point_map = scene_tensor['point_map'].permute(0, 3, 1, 2)
        images = scene_tensor['color_images'].permute(0, 3, 1, 2)
        question_type = get_sqa_question_type(question)
            
        # convert answer format
        answer_label = torch.zeros(self.num_answers).long()
        for _id in answer_id_list:
            answer_label[_id] = 1
        
        return {
            "sentence": concat_sentence,
            "scan_dir": os.path.join(self.base_dir, 'scans'),
            "scan_id": scan_id,
            "answer_label": answer_label, # A  
            "point_map" : point_map,
            "images": images,
            "sqa_type": question_type
        }

    def build_answer(self):
        """Build the SQA3D answer vocabulary."""
        answer_data = json.load(
            open(os.path.join(self.base_dir,
                              'annotations/sqa3d/answer_dict.json'), encoding='utf-8')
            )[0]
        answer_counter = []
        for data in answer_data.keys():
            answer_counter.append(data)
        answer_counter = collections.Counter(sorted(answer_counter))
        num_answers = len(answer_counter)
        answer_cands = answer_counter.keys()
        answer_vocab = SQA3DAnswer(answer_cands)
        return num_answers, answer_vocab, answer_cands

    def _load_lang(self):
        """Load SQA3D annotations for the current split."""
        lang_data = []
        scan_ids = set()
        anno_file = os.path.join(self.base_dir,
            f'annotations/sqa3d/balanced/v1_balanced_sqa_annotations_{self.split}_scannetv2.json')
        json_data = json.load(open(anno_file, 'r', encoding='utf-8'))['annotations']
        
        for item in json_data:
            if self.use_unanswer or (len(set(item['answers']) & set(self.answer_cands)) > 0):
                scan_ids.add(item['scene_id'])
                lang_data.append(item)
        print(f'{self.split} unanswerable question {len(json_data) - len(lang_data)},'
              + f'answerable question {len(lang_data)}')

        return lang_data, scan_ids

    def _load_question(self):
        """Load SQA3D question metadata keyed by scan and question id."""
        questions_map = {}
        anno_file = os.path.join(self.base_dir,
            f'annotations/sqa3d/balanced/v1_balanced_questions_{self.split}_scannetv2.json')
        json_data = json.load(open(anno_file, 'r', encoding='utf-8'))['questions']
        for item in json_data:
            if item['scene_id'] not in questions_map.keys():
                questions_map[item['scene_id']] = {}
            questions_map[item['scene_id']][item['question_id']] = {
                'situation': [item['situation']] + item['alternative_situation'],   # list of sentences
                'question': item['question']   # sentence
            }

        return questions_map


@DATASET_REGISTRY.register()
class ScanNetScanQA(ScanNetBase):
    """ScanNet dataset for the ScanQA task."""

    def __init__(self, cfg, split):
        """Load ScanQA annotations and scan metadata."""
        super(ScanNetScanQA, self).__init__(cfg, split)

        self.use_unanswer = cfg.data.get(self.__class__.__name__).get(split).use_unanswer

        assert cfg.data.args.sem_type in ['607']
        assert self.split in ['train', 'val', 'test']
        # TODO: hack test split to be the same as val
        if self.split == 'test':
            self.split = 'val'
            
        self.is_test = ('test' in self.split)
        print(f"Loading ScanNet ScanQA {split}-set language")
        self.num_answers, self.answer_vocab, self.answer_cands = self.build_answer()
        lang_data, self.scan_ids = self._load_lang()
        if cfg.debug.flag and cfg.debug.debug_size != -1:
            self.lang_data = []
            self.scan_ids = sorted(list(self.scan_ids))[:cfg.debug.debug_size]
            for item in lang_data:
                if item['scene_id'] in self.scan_ids:
                    self.lang_data.append(item)
        else:
            self.lang_data = lang_data
        print(f"Finish loading ScanNet ScanQA {split}-set language")

        print(f"Loading ScanNet ScanQA {split}-set scans")
        self.scan_data = self._load_scannet(self.scan_ids)
        print(f"Finish loading ScanNet ScanQA {split}-set data")

    def __getitem__(self, index):
        """Return one ScanQA sample."""
        item = self.lang_data[index]
        item_id = item['question_id']
        scan_id =  item['scene_id']
        if not self.is_test:
            answer_list = item['answers']
            answer_id_list = [self.answer_vocab.stoi(answer) 
                              for answer in answer_list if self.answer_vocab.stoi(answer) >= 0]
        else:
            answer_list = []
            answer_id_list = []
        question = item['question']
        scene_tensor = load_safetensor_from_hf(
            repo_id="MatchLab/ScenePoint",
            filename=self.scan_data[scan_id]["safetensors_path"],
        )
        point_map = scene_tensor['point_map'].permute(0, 3, 1, 2)
        images = scene_tensor['color_images'].permute(0, 3, 1, 2)

        # convert answer format
        answer_label = torch.zeros(self.num_answers)
        for _id in answer_id_list:
            answer_label[_id] = 1
            
        data_dict = {
            "sentence": question,
            "scan_dir": os.path.join(self.base_dir, 'scans'),
            "scan_id": scan_id,
            "answers": "[answer_seq]".join(answer_list),
            "answer_label": answer_label.float(),
            "point_map": point_map,
            "images": images,
            "data_idx": item_id,
        }

        return data_dict

    def _load_lang(self):
        """Load ScanQA annotations for the current split."""
        lang_data = []
        scan_ids = set()
        anno_file = os.path.join(self.base_dir,
                                 f'annotations/scanqa/ScanQA_v1.0_{self.split}.json')

        json_data = json.load(open(anno_file, 'r', encoding='utf-8'))
        for item in json_data:
            if self.use_unanswer or (len(set(item['answers']) & set(self.answer_cands)) > 0):
                scan_ids.add(item['scene_id'])
                lang_data.append(item)
        print(f'{self.split} unanswerable question {len(json_data) - len(lang_data)},'
              + f'answerable question {len(lang_data)}')
        return lang_data, scan_ids

    def build_answer(self):
        """Build the ScanQA answer vocabulary from the train set."""
        train_data = json.load(open(os.path.join(self.base_dir,
                                'annotations/scanqa/ScanQA_v1.0_train.json'), encoding='utf-8'))
        answer_counter = sum([data['answers'] for data in train_data], [])
        answer_counter = collections.Counter(sorted(answer_counter))
        num_answers = len(answer_counter)
        answer_cands = answer_counter.keys()
        answer_vocab = ScanQAAnswer(answer_cands)
        return num_answers, answer_vocab, answer_cands
