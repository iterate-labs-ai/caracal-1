"""Plot loss curve from W&B."""
import argparse
import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--last", default="24h")
    args = parser.parse_args()

    api = wandb.Api()
    run = api.run(f"iterate-labs/{args.run_id}")
    history = run.history()

    if "train/loss" in history.columns:
        print(history[["_step", "train/loss"]].tail(20).to_string())
    else:
        print("No train/loss in history")


if __name__ == "__main__":
    main()
