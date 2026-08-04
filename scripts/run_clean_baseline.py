import argparse

from scripts.common import load_config, run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run clean CE baseline")
    parser.add_argument("--config", default="configs/debug.yaml")
    parser.add_argument("--ordinary-augmentation", action="store_true")
    args = parser.parse_args()
    run_experiment(load_config(args.config), mode="ce_aug" if args.ordinary_augmentation else "ce", degrade=False)
