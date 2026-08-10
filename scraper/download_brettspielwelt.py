"""Download raw Brettspielwelt Tichu logs by numeric game id.

The ``.tch`` endpoints return plain text, so BeautifulSoup is intentionally not
used here.  It is only needed for parsing the optional monthly HTML indexes.
"""

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://tichulog.brettspielwelt.de"
DEFAULT_LATEST_ID = 2_419_834
REQUIRED_MARKERS = (
    "---------------Gr.Tichukarten------------------",
    "---------------Startkarten------------------",
    "Schupfen:",
    "---------------Rundenverlauf------------------",
)


def classify_log(text):
    """Classify a response without discarding potentially useful raw logs."""
    if not all(marker in text for marker in REQUIRED_MARKERS):
        return "invalid"
    if "Ergebnis:" not in text:
        return "incomplete"
    return "complete"


def game_ids(*, latest_id, count, start_id=None, end_id=None):
    """Return an inclusive ascending id range."""
    if (start_id is None) != (end_id is None):
        raise ValueError("--start-id and --end-id must be supplied together")
    if start_id is not None:
        if start_id < 1 or end_id < start_id:
            raise ValueError("invalid explicit id range")
        return range(start_id, end_id + 1)
    if latest_id < 1 or count < 1:
        raise ValueError("--latest-id and --count must be positive")
    return range(max(1, latest_id - count + 1), latest_id + 1)


def _fetch(url, *, timeout):
    request = Request(
        url,
        headers={"User-Agent": "Tichu-research-downloader/1.0"},
    )
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def download_game(
    game_id,
    *,
    output_dir,
    base_url=DEFAULT_BASE_URL,
    timeout=20.0,
    retries=2,
    overwrite=False,
    fetch=_fetch,
):
    """Download one game atomically and return a manifest record."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "{}.tch".format(game_id)
    now = datetime.now(timezone.utc).isoformat()

    if destination.exists() and not overwrite:
        data = destination.read_bytes()
        text = data.decode("utf-8", errors="replace")
        return {
            "game_id": game_id,
            "status": "skipped_{}".format(classify_log(text)),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "url": "{}/{}.tch".format(base_url.rstrip("/"), game_id),
            "timestamp": now,
        }

    url = "{}/{}.tch".format(base_url.rstrip("/"), game_id)
    last_error = None
    for attempt in range(retries + 1):
        try:
            data = fetch(url, timeout=timeout)
            text = data.decode("utf-8", errors="replace")
            status = classify_log(text)
            if status == "invalid":
                return {
                    "game_id": game_id,
                    "status": status,
                    "bytes": len(data),
                    "url": url,
                    "timestamp": now,
                }

            temporary = destination.with_suffix(".tch.part")
            temporary.write_bytes(data)
            os.replace(temporary, destination)
            return {
                "game_id": game_id,
                "status": status,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "url": url,
                "timestamp": now,
            }
        except HTTPError as error:
            last_error = error
            if error.code < 500 or attempt == retries:
                break
        except (URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt == retries:
                break

    return {
        "game_id": game_id,
        "status": "http_{}".format(last_error.code)
        if isinstance(last_error, HTTPError)
        else "network_error",
        "error": str(last_error),
        "url": url,
        "timestamp": now,
    }


def append_manifest(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        output.write("\n")


def run(args):
    ids = game_ids(
        latest_id=args.latest_id,
        count=args.count,
        start_id=args.start_id,
        end_id=args.end_id,
    )
    output_dir = Path(args.output_dir)
    manifest = output_dir / "manifest.jsonl"
    counts = Counter()

    for index, game_id in enumerate(ids, start=1):
        record = download_game(
            game_id,
            output_dir=output_dir,
            base_url=args.base_url,
            timeout=args.timeout,
            retries=args.retries,
            overwrite=args.overwrite,
        )
        append_manifest(manifest, record)
        counts[record["status"]] += 1
        if index == 1 or index % args.progress_every == 0:
            print(
                "downloaded {}/{} through id {}: {}".format(
                    index, len(ids), game_id, dict(sorted(counts.items()))
                ),
                flush=True,
            )
        if args.delay and index < len(ids):
            time.sleep(args.delay)

    summary = {
        "start_id": ids.start,
        "end_id": ids.stop - 1,
        "requested": len(ids),
        "counts": dict(sorted(counts.items())),
        "output_dir": str(output_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description="Download raw Brettspielwelt .tch logs with resume support."
    )
    parser.add_argument("--latest-id", type=int, default=DEFAULT_LATEST_ID)
    parser.add_argument("--count", type=int, default=1_000)
    parser.add_argument("--start-id", type=int)
    parser.add_argument("--end-id", type=int)
    parser.add_argument(
        "--output-dir", default="datasets/raw/brettspielwelt"
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.delay < 0 or args.timeout <= 0 or args.retries < 0:
        raise SystemExit("delay/retries must be non-negative and timeout positive")
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be positive")
    run(args)


if __name__ == "__main__":
    main()
