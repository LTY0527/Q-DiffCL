import argparse

from scripts.common import load_config, run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/debug.yaml")
    parser.add_argument("--mode", choices=["supcon", "joint", "linear_probe", "fine_tune"], default="joint")
    args = parser.parse_args()
    run_experiment(load_config(args.config), mode=args.mode, degrade=True)
