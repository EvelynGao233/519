"""Upload hf_upload/dataset and hf_upload/model to the Hugging Face Hub.

Run `hf auth login` yourself first; this script only uses the saved login.

  python src/upload_hf.py --dataset-name fox-nbc-headline-links --model-name fox-nbc-headline-classifier [--private]
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-name", default="fox-nbc-headline-links")
    p.add_argument("--model-name", default="fox-nbc-headline-classifier")
    p.add_argument("--private", action="store_true")
    a = p.parse_args()

    api = HfApi()
    user = api.whoami()["name"]
    ds_id, model_id = f"{user}/{a.dataset_name}", f"{user}/{a.model_name}"

    card = ROOT / "hf_upload/model/README.md"
    card.write_text(card.read_text(encoding="utf-8").replace("[DATASET_REPO]", ds_id), encoding="utf-8")

    for repo_id, kind, folder in [(ds_id, "dataset", "hf_upload/dataset"), (model_id, "model", "hf_upload/model")]:
        api.create_repo(repo_id, repo_type=kind, private=a.private, exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type=kind, folder_path=str(ROOT / folder),
                          commit_message="Upload from CIS 5190 Track B project")
        prefix = "datasets/" if kind == "dataset" else ""
        print(f"{kind}: https://huggingface.co/{prefix}{repo_id}")


if __name__ == "__main__":
    main()
