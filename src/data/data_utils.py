"""Dataset-side helper functions and answer vocab classes."""

import re
from functools import lru_cache
from pathlib import Path

from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE, HF_HUB_OFFLINE
from huggingface_hub.errors import LocalEntryNotFoundError


@lru_cache(maxsize=8192)
def _find_cached_hf_file(repo_id, filename, repo_type="dataset"):
    """Find a cached Hugging Face file without downloading it again."""
    repo_prefix = {
        "dataset": "datasets",
        "model": "models",
        "space": "spaces",
    }.get(repo_type, f"{repo_type}s")
    repo_cache_dir = Path(HF_HUB_CACHE) / f"{repo_prefix}--{repo_id.replace('/', '--')}"
    snapshots_dir = repo_cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    matches = [path for path in snapshots_dir.glob(f"*/{filename}") if path.is_file()]
    if not matches:
        return None

    matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(matches[0])


def load_safetensor_from_hf(repo_id, filename, repo_type="dataset"):
    """Load a safetensor file from Hugging Face or the local cache."""
    if HF_HUB_OFFLINE:
        cached_path = _find_cached_hf_file(repo_id=repo_id, filename=filename, repo_type=repo_type)
        if cached_path is None:
            raise LocalEntryNotFoundError(
                f"Cannot find {filename} in the local Hugging Face cache for repo {repo_id}."
            )
        return load_file(cached_path)

    try:
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_files_only=False,
        )
    except LocalEntryNotFoundError:
        cached_path = _find_cached_hf_file(repo_id=repo_id, filename=filename, repo_type=repo_type)
        if cached_path is None:
            raise
    return load_file(cached_path)


class ScanQAAnswer(object):
    """Answer vocabulary wrapper for ScanQA."""

    def __init__(self, answers=None, unk_token='<unk>', ignore_idx=-100):
        """Build the ScanQA answer vocabulary."""
        if answers is None:
            answers = []
        self.unk_token = unk_token
        self.ignore_idx = ignore_idx
        self.vocab = {x: i for i, x in enumerate(answers)}
        self.rev_vocab = dict((v, k) for k, v in self.vocab.items())

    def itos(self, i):
        """Convert an answer id to text."""
        if i == self.ignore_idx:
            return self.unk_token
        return self.rev_vocab[i]

    def stoi(self, v):
        """Convert answer text to an id."""
        if v not in self.vocab:
            return self.ignore_idx
        return self.vocab[v]

    def __len__(self):
        """Return the number of answers in the vocabulary."""
        return len(self.vocab)


class Hypo3DAnswer(object):
    """Answer vocabulary wrapper for Hypo3D."""

    def __init__(self, answers=None):
        """Build the Hypo3D answer vocabulary."""
        if answers is None:
            answers = []
        self.vocab = {x: i for i, x in enumerate(answers)}
        self.rev_vocab = dict((v, k) for k, v in self.vocab.items())

    def itos(self, i):
        """Convert an answer id to text."""
        return self.rev_vocab[i]

    def stoi(self, v):
        """Convert answer text to an id."""
        return self.vocab[v]

    def __len__(self):
        """Return the number of answers in the vocabulary."""
        return len(self.vocab)


class msnnAnswer(object):
    """Answer vocabulary wrapper for MSNN."""

    def __init__(self, answers=None):
        """Build the MSNN answer vocabulary."""
        if answers is None:
            answers = []
        self.vocab = {x: i for i, x in enumerate(answers)}
        self.rev_vocab = dict((v, k) for k, v in self.vocab.items())

    def itos(self, i):
        """Convert an answer id to text."""
        return self.rev_vocab[i]

    def stoi(self, v):
        """Convert answer text to an id."""
        return self.vocab[v]

    def __len__(self):
        """Return the number of answers in the vocabulary."""
        return len(self.vocab)


class SQA3DAnswer(object):
    """Answer vocabulary wrapper for SQA3D."""

    def __init__(self, answers=None, unk_token='u'):
        """Build the SQA3D answer vocabulary."""
        if answers is None:
            answers = []
        self.vocab = {x: i for i, x in enumerate(answers)}
        self.rev_vocab = dict((v, k) for k, v in self.vocab.items())
        self.unk_token = unk_token
        self.ignore_idx = self.vocab['u']

    def itos(self, i):
        """Convert an answer id to text."""
        if i == self.ignore_idx:
            return self.unk_token
        return self.rev_vocab[i]

    def stoi(self, v):
        """Convert answer text to an id."""
        if v not in self.vocab:
            return self.ignore_idx
        return self.vocab[v]

    def __len__(self):
        """Return the number of answers in the vocabulary."""
        return len(self.vocab)


def get_sqa_question_type(question):
    """Map an SQA question prefix to a small question-type id."""
    question = question.lstrip()
    if question[:4].lower() == 'what':
        return 0
    elif question[:2].lower() == 'is':
        return 1
    elif question[:3].lower() == 'how':
        return 2
    elif question[:3].lower() == 'can':
        return 3
    elif question[:5].lower() == 'which':
        return 4
    else:
        return 5


def clean_answer(data):
    """Normalize free-form answers before comparison."""
    data = data.lower()
    data = re.sub(r'[ ]+$', '', data)
    data = re.sub(r'^[ ]+', '', data)
    data = re.sub(r' {2,}', ' ', data)

    data = re.sub(r'\.[ ]{2,}', '. ', data)
    data = re.sub(r"[^a-zA-Z0-9,'\s\-:]+", '', data)
    data = re.sub(r'ç', 'c', data)
    data = re.sub(r'’', "'", data)
    data = re.sub(r'\bletf\b', 'left', data)
    data = re.sub(r'\blet\b', 'left', data)
    data = re.sub(r'\btehre\b', 'there', data)
    data = re.sub(r'\brigth\b', 'right', data)
    data = re.sub(r'\brght\b', 'right', data)
    data = re.sub(r'\bbehine\b', 'behind', data)
    data = re.sub(r'\btv\b', 'TV', data)
    data = re.sub(r'\bchai\b', 'chair', data)
    data = re.sub(r'\bwasing\b', 'washing', data)
    data = re.sub(r'\bwaslked\b', 'walked', data)
    data = re.sub(r'\boclock\b', "o'clock", data)
    data = re.sub(r"\bo'[ ]+clock\b", "o'clock", data)

    data = re.sub(r'\b0\b', 'zero', data)
    data = re.sub(r'\bnone\b', 'zero', data)
    data = re.sub(r'\b1\b', 'one', data)
    data = re.sub(r'\b2\b', 'two', data)
    data = re.sub(r'\b3\b', 'three', data)
    data = re.sub(r'\b4\b', 'four', data)
    data = re.sub(r'\b5\b', 'five', data)
    data = re.sub(r'\b6\b', 'six', data)
    data = re.sub(r'\b7\b', 'seven', data)
    data = re.sub(r'\b8\b', 'eight', data)
    data = re.sub(r'\b9\b', 'nine', data)
    data = re.sub(r'\b10\b', 'ten', data)
    data = re.sub(r'\b11\b', 'eleven', data)
    data = re.sub(r'\b12\b', 'twelve', data)
    data = re.sub(r'\b13\b', 'thirteen', data)
    data = re.sub(r'\b14\b', 'fourteen', data)
    data = re.sub(r'\b15\b', 'fifteen', data)
    data = re.sub(r'\b16\b', 'sixteen', data)
    data = re.sub(r'\b17\b', 'seventeen', data)
    data = re.sub(r'\b18\b', 'eighteen', data)
    data = re.sub(r'\b19\b', 'nineteen', data)
    data = re.sub(r'\b20\b', 'twenty', data)
    data = re.sub(r'\b23\b', 'twenty-three', data)

    data = re.sub(r'\b([a-zA-Z]+)([0-9])\b', r'\g<1>', data)
    data = re.sub(r'\ba\b ([a-zA-Z]+)', r'\g<1>', data)
    data = re.sub(r'\ban\b ([a-zA-Z]+)', r'\g<1>', data)
    data = re.sub(r'\bthe\b ([a-zA-Z]+)', r'\g<1>', data)
    data = re.sub(r'\bbackwards\b', 'backward', data)

    return data
