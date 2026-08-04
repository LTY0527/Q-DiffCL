from scripts.common import load_config, parser, run_experiment

if __name__ == "__main__":
    args = parser("Evaluate CE under configured degradation").parse_args()
    run_experiment(load_config(args.config), mode="ce", degrade=True)

