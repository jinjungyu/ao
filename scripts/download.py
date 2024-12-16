# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# copied from https://github.com/pytorch-labs/gpt-fast/blob/main/scripts/download.py
import os
from typing import Optional

from requests.exceptions import HTTPError
from pathlib import Path

def hf_download(repo_id: Optional[str] = None, save_dir: Optional[str] = None, hf_token: Optional[str] = None) -> None:
    from huggingface_hub import snapshot_download
    local_dir = Path(save_dir)/repo_id
    os.makedirs(local_dir, exist_ok=True)

    try:
        snapshot_download(repo_id, local_dir=local_dir, local_dir_use_symlinks=False, token=hf_token)
        print(f"Model is successfully downloaded to {local_dir}")
    except HTTPError as e:
        if e.response.status_code == 401:
            print("You need to pass a valid `--hf_token=...` to download private checkpoints.")
        else:
            raise e

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download data from HuggingFace Hub.')
    parser.add_argument('--repo_id', type=str, default="meta-llama/Llama-2-7b-chat-hf", help='Repository ID to download from.')
    parser.add_argument('--save_dir', type=str, default="checkpoints")
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace API token.')

    args = parser.parse_args()
    hf_download(args.repo_id, args.save_dir, args.hf_token)
