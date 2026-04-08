"""
Download and iterate over the Aerial234 dataset from HuggingFace.

Dataset: RussRobin/Aerial234
  - 234 aerial images from continuous UAV scan of Southeast University
  - Used for image stitching research (AAAI 2025, JVCI 2023/2025)

Images are downloaded to a local cache and paths are yielded in sequence order.

Authentication:
  The dataset is gated. Log in once with:
    python -c "from huggingface_hub import login; login()"
  Or set the HF_TOKEN environment variable to your token.
  Get a token at: https://huggingface.co/settings/tokens
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, Optional

from huggingface_hub import snapshot_download

DATASET_REPO = "RussRobin/Aerial234"
DEFAULT_CACHE = Path("data/aerial234")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def _resolve_token(token: Optional[str]) -> Optional[str]:
    """Return token from argument or HF_TOKEN env var."""
    return token or os.environ.get("HF_TOKEN") or None


def download(cache_dir: str | Path = DEFAULT_CACHE, token: Optional[str] = None) -> Path:
    """
    Download the Aerial234 dataset to a local cache directory.

    Args:
        cache_dir: Local directory to store downloaded files.
        token: HuggingFace access token. Falls back to HF_TOKEN env var,
               then to the cached token from `hf auth login`.

    Returns:
        Local path to the dataset root.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    hf_token = _resolve_token(token)
    print(f"[dataset] Downloading {DATASET_REPO} -> {cache_dir} ...")
    if hf_token:
        print("[dataset] Using provided HF token.")
    else:
        print("[dataset] No token provided; using cached login (if any).")

    try:
        local_dir = snapshot_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            local_dir=str(cache_dir),
            token=hf_token,
        )
    except Exception as exc:
        if "401" in str(exc) or "Unauthorized" in str(exc):
            raise RuntimeError(
                "\n[dataset] Authentication required for RussRobin/Aerial234.\n"
                "  Option 1: Run once to cache credentials:\n"
                "    python -c \"from huggingface_hub import login; login()\"\n"
                "  Option 2: Set environment variable:\n"
                "    set HF_TOKEN=hf_...\n"
                "  Option 3: Pass --hf-token to the pipeline:\n"
                "    python pipeline.py --hf-token hf_...\n"
                "  Get your token at: https://huggingface.co/settings/tokens"
            ) from exc
        raise

    print(f"[dataset] Download complete: {local_dir}")
    return Path(local_dir)


def iter_images(
    dataset_dir: str | Path,
    extensions: tuple[str, ...] = IMAGE_EXTENSIONS,
) -> Generator[Path, None, None]:
    """
    Yield image paths from the dataset directory in sorted (sequence) order.

    Sorting by filename preserves the drone flight sequence order.
    """
    dataset_dir = Path(dataset_dir)
    images = sorted(
        p for p in dataset_dir.rglob("*")
        if p.suffix.lower() in extensions and not p.name.startswith(".")
    )
    if not images:
        raise FileNotFoundError(
            f"No images found in {dataset_dir}. "
            "Run download() first or check the cache path."
        )
    return (p for p in images)


def load_or_download(
    cache_dir: str | Path = DEFAULT_CACHE,
    token: Optional[str] = None,
) -> list[Path]:
    """
    Return sorted list of image paths, downloading if needed.
    """
    cache_dir = Path(cache_dir)
    images = sorted(
        p for p in cache_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith(".")
    )
    if not images:
        root = download(cache_dir, token=token)
        images = sorted(
            p for p in root.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith(".")
        )
    return images
