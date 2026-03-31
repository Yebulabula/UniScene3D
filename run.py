"""Main training and evaluation entry point."""

from datetime import datetime
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import hydra
from omegaconf import OmegaConf, open_dict
import wandb

from common.misc import make_dir, rgetattr
from trainer.build import build_trainer


@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg):
    if cfg.resume:
        assert Path(cfg.exp_dir).exists(), f"Resuming failed: {cfg.exp_dir} does not exist."
        print(f"Resuming from {cfg.exp_dir}")
        cfg = OmegaConf.load(Path(cfg.exp_dir) / 'config.yaml')
        cfg.resume = True
    else:
        run_id = wandb.util.generate_id()
        with open_dict(cfg):
            cfg.logger.run_id = run_id

    OmegaConf.resolve(cfg)
    naming_keys = [cfg.name]
    for name in cfg.get('naming_keywords', []):
        if name == "time":
            continue
        elif name == "task":
            naming_keys.append(cfg.task)
            if rgetattr(cfg, "data.note", None) is not None:
                naming_keys.append(rgetattr(cfg, "data.note"))
            else:
                datasets = rgetattr(cfg, "data.train")
                dataset_names = "+".join([str(x) for x in datasets])
                naming_keys.append(dataset_names)
        elif name == "dataloader.batchsize":
            naming_keys.append(f"b{rgetattr(cfg, name) * rgetattr(cfg, 'num_gpu')}")
        else:
            if str(rgetattr(cfg, name)) != "":
                naming_keys.append(str(rgetattr(cfg, name)))
    exp_name = "_".join(naming_keys)

    if rgetattr(cfg, "debug.flag", False):
        exp_name = "Debug_test"
    print(exp_name)

    if not cfg.exp_dir:
        cfg.exp_dir = Path(cfg.base_dir) / exp_name / f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f')}"
    else:
        cfg.exp_dir = Path(cfg.exp_dir)
    make_dir(cfg.exp_dir)
    OmegaConf.save(cfg, cfg.exp_dir / "config.yaml")

    trainer = build_trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
