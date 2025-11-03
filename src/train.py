import argparse
import yaml
from pathlib import Path


def load_config(path: str):
    """
    Load YAML config from a file.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("Loaded config:", cfg)

    # TODO: integrate with external/LaneATT
    # e.g. import external.LaneATT.train as laneatt_train
    # laneatt_train.train(cfg)


if __name__ == "__main__":
    main()