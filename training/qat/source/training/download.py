"""Pinned, checksummed downloads; no full repository/model clone required."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import requests

DATA_REV = "cf95b1bc2fc90c32d3b5455f627a20bc43fd3db7"
TOKENIZER_REV = "8c0c0de640cd532fee03d329b95a58c56591b5bc"
TOKENIZER_FILES = {
    "tokenizer.json": (439547, "d98595c6aef70d95f72748582fb9b4f53d76dd58c1ae1dd702ad7c84e1caf5e4"),
    "tokenizer_config.json": (1029, "9e5a36a68712591f8c63fef9874b5c4af20b87e9ca67ba28d0b61e5cb2a133de"),
}
DATASETS = {
    "pretrain": ("pretrain_hq.jsonl", 1669750047,
                 "9801b0d2210c61c2e4bc130f6dc4b3c870698a88d04af8f103c23dd5f0ce2440"),
    "sft": ("sft_mini_512.jsonl", 1232540940,
            "475039fa9b80ad365bc23d837b1a7695c7c680d950713a91359621fbb7607852"),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url, path, size=None, digest=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if (size is None or path.stat().st_size == size) and (digest is None or sha256(path) == digest):
            print(f"Already verified: {path}", flush=True)
            return
        raise ValueError(f"Existing file failed validation: {path}; move it aside before retrying")
    part = path.with_name(path.name + ".part")
    for attempt in range(6):
        offset = part.stat().st_size if part.exists() else 0
        try:
            if size is not None and offset == size:
                break
            with requests.get(url, headers={"Range": f"bytes={offset}-"} if offset else {},
                              stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Server returned an unexpected byte range")
                elif response.status_code == 200:
                    offset = 0  # A server may ignore Range; start over rather than append corrupt data.
                else:
                    raise ValueError(f"Unexpected download status {response.status_code}")
                start = time.monotonic()
                last = start
                received = 0
                with open(part, "ab" if offset else "wb") as stream:
                    for chunk in response.iter_content(1 << 20):
                        stream.write(chunk)
                        received += len(chunk)
                        now = time.monotonic()
                        if now - last > 5:
                            print(f"{path.name}: {(offset+received)/1e6:.1f} MB, "
                                  f"{received/max(now-start, .001)/1e6:.1f} MB/s", flush=True)
                            last = now
            if size is not None and part.stat().st_size != size:
                raise OSError("Incomplete download")
            break
        except (requests.RequestException, OSError) as exc:
            if attempt == 5:
                raise
            print(f"Retry {attempt+1}: {exc}", flush=True)
            time.sleep(min(2 ** attempt, 15))
    if size is not None and part.stat().st_size != size:
        raise ValueError(f"Wrong download size: {part}")
    if digest is not None and sha256(part) != digest:
        raise ValueError(f"SHA256 mismatch: {part}; move this partial file aside to redownload")
    os.replace(part, path)
    print(f"Saved: {path} ({path.stat().st_size:,} bytes)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("training-data/raw"))
    parser.add_argument("--dataset", choices=["pretrain", "sft", "none"], default="pretrain")
    parser.add_argument("--sample-docs", type=int, default=0,
                        help="Download only a small real-data smoke sample; NOT a full verified corpus")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = args.out / "tokenizer"
    for name, (size, digest) in TOKENIZER_FILES.items():
        # Resolve a pinned git blob; requests enforces TLS certificate checks.
        fetch(f"https://huggingface.co/jingyaogong/MiniMind2-Small/resolve/{TOKENIZER_REV}/{name}",
              tokenizer_dir / name, size, digest)
    provenance = {"dataset_revision": DATA_REV, "tokenizer_revision": TOKENIZER_REV,
                  "tokenizer_sha256": sha256(tokenizer_dir / "tokenizer.json"),
                  "sample_docs": args.sample_docs}
    if args.dataset != "none":
        name, size, digest = DATASETS[args.dataset]
        url = f"https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/{DATA_REV}/{name}"
        provenance.update(url=url, full_size=size, full_sha256=digest)
        if args.sample_docs:
            if args.sample_docs < 1:
                parser.error("sample-docs must be positive")
            path = args.out / ("sample-" + name)
            part = path.with_suffix(".jsonl.part")
            count = 0
            with requests.get(url, stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                with open(part, "wb") as stream:
                    for line in response.iter_lines(chunk_size=65536):
                        if not line.strip():
                            continue
                        json.loads(line)
                        stream.write(line + b"\n")
                        count += 1
                        if count >= args.sample_docs:
                            break
            if count != args.sample_docs:
                raise ValueError("Sample shorter than requested")
            os.replace(part, path)
            provenance.update(path=str(path), sample_sha256=sha256(path))
            print(f"Saved {count} sample documents: {path}", flush=True)
        else:
            path = args.out / name
            fetch(url, path, size, digest)
            provenance["path"] = str(path)
    (args.out / f"{args.dataset}{'-sample' if args.sample_docs else ''}-source.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
