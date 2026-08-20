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
from concurrent.futures import ThreadPoolExecutor
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


def _is_log_response(game_id, *, base_url, timeout, retries, fetch):
    """Return whether an id serves a BrettspielWelt log without writing it."""
    url = "{}/{}.tch".format(base_url.rstrip("/"), game_id)
    for attempt in range(retries + 1):
        try:
            data = fetch(url, timeout=timeout)
            text = data.decode("utf-8", errors="replace")
            return classify_log(text) != "invalid"
        except HTTPError as error:
            if error.code < 500 or attempt == retries:
                return False
        except (URLError, TimeoutError, OSError):
            if attempt == retries:
                return False
    return False


def discover_latest_id(
    *,
    hint=DEFAULT_LATEST_ID,
    base_url=DEFAULT_BASE_URL,
    timeout=20.0,
    retries=2,
    initial_step=256,
    fetch=None,
):
    """Find the newest contiguous log id using exponential and binary search.

    BrettspielWelt returns an HTTP 500 response for ids beyond the current
    boundary.  A previously known valid id is therefore enough to locate the
    new boundary in logarithmic requests.
    """
    if hint < 1 or initial_step < 1:
        raise ValueError("hint and initial_step must be positive")
    fetch = fetch or _fetch
    probe = lambda game_id: _is_log_response(
        game_id,
        base_url=base_url,
        timeout=timeout,
        retries=retries,
        fetch=fetch,
    )
    if not probe(hint):
        raise ValueError("latest-id hint {} is not a valid log".format(hint))

    low = hint
    step = initial_step
    high = low + step
    while probe(high):
        low = high
        step *= 2
        high = low + step

    while high - low > 1:
        middle = (low + high) // 2
        if probe(middle):
            low = middle
        else:
            high = middle
    return low


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


def count_complete_logs(output_dir):
    """Count complete cached logs, ignoring manifests and partial files."""
    complete = 0
    for path in Path(output_dir).glob("*.tch"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if classify_log(text) == "complete":
            complete += 1
    return complete


def collect_latest(
    *,
    latest_id,
    target_total,
    output_dir,
    base_url=DEFAULT_BASE_URL,
    delay=0.1,
    timeout=20.0,
    retries=2,
    overwrite=False,
    progress_every=100,
    max_consecutive_failures=500,
    workers=1,
    fetch=_fetch,
    sleep=time.sleep,
):
    """Collect newest-first until the cache contains target_total complete logs."""
    if latest_id < 1 or target_total < 1:
        raise ValueError("latest_id and target_total must be positive")
    if progress_every < 1 or max_consecutive_failures < 1 or workers < 1:
        raise ValueError("progress/failure limits must be positive")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.jsonl"
    complete = count_complete_logs(output_dir)
    initial_complete = complete
    counts = Counter()
    scanned = 0
    consecutive_failures = 0
    last_id = None

    ids = iter(range(latest_id, 0, -1))
    stopped_for_failures = False
    with ThreadPoolExecutor(max_workers=workers) as executor:
        while complete < target_total:
            batch_size = min(workers, target_total - complete)
            batch_ids = []
            for _ in range(batch_size):
                try:
                    batch_ids.append(next(ids))
                except StopIteration:
                    break
            if not batch_ids:
                break

            futures = [
                executor.submit(
                    download_game,
                    game_id,
                    output_dir=output_dir,
                    base_url=base_url,
                    timeout=timeout,
                    retries=retries,
                    overwrite=overwrite,
                    fetch=fetch,
                )
                for game_id in batch_ids
            ]
            records = [future.result() for future in futures]
            made_request = False
            for game_id, record in zip(batch_ids, records):
                append_manifest(manifest, record)
                status = record["status"]
                counts[status] += 1
                scanned += 1
                last_id = game_id
                made_request = made_request or not status.startswith("skipped_")

                if status == "complete":
                    complete += 1
                    consecutive_failures = 0
                elif status in {
                    "skipped_complete",
                    "incomplete",
                    "skipped_incomplete",
                }:
                    # Cached complete logs were included in initial_complete.
                    # Existing incomplete logs still prove that the id exists.
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1

                if (
                    scanned == 1
                    or scanned % progress_every == 0
                    or complete >= target_total
                ):
                    percent = 100.0 * complete / target_total
                    print(
                        "[download] complete {}/{} ({:.1f}%), scanned {}, id {}: {}".format(
                            complete,
                            target_total,
                            min(100.0, percent),
                            scanned,
                            game_id,
                            dict(sorted(counts.items())),
                        ),
                        flush=True,
                    )

            if consecutive_failures >= max_consecutive_failures:
                print(
                    "[download] stopped after {} consecutive failures at id {}".format(
                        consecutive_failures, last_id
                    ),
                    flush=True,
                )
                stopped_for_failures = True
                break
            if delay and made_request and complete < target_total:
                sleep(delay)

    summary = {
        "latest_id": latest_id,
        "last_scanned_id": last_id,
        "target_total": target_total,
        "initial_complete": initial_complete,
        "complete": complete,
        "new_complete": complete - initial_complete,
        "scanned": scanned,
        "workers": workers,
        "stopped_for_failures": stopped_for_failures,
        "reached_target": complete >= target_total,
        "counts": dict(sorted(counts.items())),
        "output_dir": str(output_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return summary


def run(args):
    latest_id = args.latest_id
    if args.discover_latest:
        print(
            "[download] discovering latest id from hint {}".format(latest_id),
            flush=True,
        )
        latest_id = discover_latest_id(
            hint=latest_id,
            base_url=args.base_url,
            timeout=args.timeout,
            retries=args.retries,
        )
        print("[download] latest id: {}".format(latest_id), flush=True)

    if args.target_total is not None:
        if args.start_id is not None or args.end_id is not None:
            raise ValueError("--target-total cannot be combined with an explicit range")
        return collect_latest(
            latest_id=latest_id,
            target_total=args.target_total,
            output_dir=args.output_dir,
            base_url=args.base_url,
            delay=args.delay,
            timeout=args.timeout,
            retries=args.retries,
            overwrite=args.overwrite,
            progress_every=args.progress_every,
            max_consecutive_failures=args.max_consecutive_failures,
            workers=args.workers,
        )

    ids = game_ids(
        latest_id=latest_id,
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
    parser.add_argument(
        "--discover-latest",
        action="store_true",
        help="discover the current last id, using --latest-id as a known-valid hint",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        help="download newest-first until this many complete logs exist",
    )
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
    parser.add_argument("--max-consecutive-failures", type=int, default=500)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="maximum concurrent downloads for --target-total",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.delay < 0 or args.timeout <= 0 or args.retries < 0:
        raise SystemExit("delay/retries must be non-negative and timeout positive")
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be positive")
    if args.target_total is not None and args.target_total < 1:
        raise SystemExit("--target-total must be positive")
    if args.max_consecutive_failures < 1:
        raise SystemExit("--max-consecutive-failures must be positive")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    run(args)


if __name__ == "__main__":
    main()
