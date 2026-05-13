"""Download TVQA-Long dataset from HuggingFace (SSL-bypass for corporate proxies)."""

import os
import httpx
from pathlib import Path
from tqdm import tqdm

BASE_URL = "https://huggingface.co/datasets/Vision-CAIR/TVQA-Long/resolve/main"
DATA_DIR = Path("data/tvqa")

FILES_TO_DOWNLOAD = {
    "annotations/tvqa_preprocessed_subtitles.json": "tvqa-long-annotations/tvqa_preprocessed_subtitles.json",
    "annotations/tvqa_val_edited.json": "tvqa-long-annotations/tvqa_val_edited.json",
    "subtitles/tvqa_subtitles.zip": "tvqa_subtitles.zip",
}

def download_file(client: httpx.Client, remote_path: str, local_path: Path):
    url = f"{BASE_URL}/{remote_path}"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        print(f"  Already exists: {local_path}")
        return

    print(f"  Downloading: {remote_path}")
    with client.stream("GET", url, follow_redirects=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(local_path, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc=local_path.name) as pbar:
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))


def download_video_part(client: httpx.Client, part: str, local_dir: Path):
    """Download a single video archive part (5GB each)."""
    remote = f"tvqa-long-videos/{part}"
    local_path = local_dir / part
    download_file(client, remote, local_path)


def main():
    print("TVQA-Long Dataset Downloader")
    print("=" * 50)

    client = httpx.Client(verify=False, timeout=httpx.Timeout(300.0), follow_redirects=True)

    # Download annotations and subtitles
    print("\n[1/2] Downloading annotations and subtitles...")
    for local_rel, remote_rel in FILES_TO_DOWNLOAD.items():
        local_path = DATA_DIR / local_rel
        download_file(client, remote_rel, local_path)

    print("\n[2/2] Video archives (optional - 52.5 GB total)")
    print("  To download videos, run:")
    print("    python download_tvqa.py --videos")
    print("  Or download specific parts:")
    print("    python download_tvqa.py --video-part aa")

    if "--videos" in os.sys.argv:
        video_dir = DATA_DIR / "videos"
        video_dir.mkdir(exist_ok=True)
        parts = [f"archive.tar.gz.{chr(97+i)}{chr(97+j)}"
                 for i in range(0, 1) for j in range(0, 11)]
        parts = [f"archive.tar.gz.a{chr(97+i)}" for i in range(11)]
        for part in parts:
            download_video_part(client, part, video_dir)
        print("\n  All video parts downloaded. Combine with:")
        print(f"    cat {video_dir}/archive.tar.gz.* | tar xzf -")

    elif "--video-part" in os.sys.argv:
        idx = os.sys.argv.index("--video-part")
        suffix = os.sys.argv[idx + 1]
        video_dir = DATA_DIR / "videos"
        video_dir.mkdir(exist_ok=True)
        download_video_part(client, f"archive.tar.gz.a{suffix}", video_dir)

    client.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
