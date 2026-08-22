"""Build a reproducible rules-v2 dataset from verified Brettspielwelt rounds."""

import argparse
import gzip
import hashlib
import io
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from gym_tichu.envs.internals import PassBombAction, PlayBomb
from gym_agents.training_data import encode_action, encode_observation
from scraper.brettspielwelt_parser import parse_file
from scraper.brettspielwelt_replay import canonical_action, replay_round


SCHEMA_VERSION = 2
RULES_VERSION = "rules-v2"
SPLIT_NAMESPACE = "brettspielwelt-decisions-v2"
SPLITS = ("train", "validation", "test")


def _validate_fractions(validation_fraction, test_fraction):
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if not 0 <= test_fraction < 1:
        raise ValueError("test_fraction must be in [0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than 1")


def _game_digest(game_id):
    return hashlib.sha256(
        "{}:{}".format(SPLIT_NAMESPACE, game_id).encode("ascii")
    ).digest()


def anonymized_game_id(game_id):
    """Return a stable pseudonym that does not expose the downloadable log ID."""
    return "bsw-{}".format(_game_digest(game_id).hex()[:16])


def split_for_game(game_id, validation_fraction=0.1, test_fraction=0.1):
    """Assign every round from one source game to one deterministic split."""
    _validate_fractions(validation_fraction, test_fraction)
    bucket = int.from_bytes(_game_digest(game_id)[:8], "big") / float(2**64)
    if bucket < test_fraction:
        return "test"
    if bucket < test_fraction + validation_fraction:
        return "validation"
    return "train"


def winning_team(scores):
    if scores[0] == scores[1]:
        return None
    return 0 if scores[0] > scores[1] else 1


def _decision_source(event, action):
    if event is None:
        if not isinstance(action, PassBombAction):
            raise ValueError("only an inferred bomb decline may omit a log event")
        return "inferred_bomb_pass"
    return "logged"


def make_record(
    *,
    game_id,
    round_index,
    decision_index,
    round_,
    state,
    event,
    action,
):
    observer = action.player_pos
    canonical = canonical_action(state, action)
    if canonical is None:
        raise ValueError("human action has no canonical legal action")
    team_points = round_.scores[observer % 2]
    opponent_points = round_.scores[(observer + 1) % 2]
    point_diff = team_points - opponent_points
    result = "win" if point_diff > 0 else "loss" if point_diff < 0 else "draw"
    legal_actions = [
        encode_action(legal_action, observer)
        for legal_action in sorted(state.possible_actions_list, key=repr)
    ]
    chosen_action = encode_action(canonical, observer)
    return {
        "schema_version": SCHEMA_VERSION,
        "rules_version": RULES_VERSION,
        "source": "brettspielwelt",
        "game_id": anonymized_game_id(game_id),
        "round": round_index,
        "decision": decision_index,
        "player": observer,
        "agent": "human",
        "decision_source": _decision_source(event, action),
        "observation": encode_observation(state, observer),
        "legal_actions": legal_actions,
        "chosen_action": chosen_action,
        "original_chosen_action": encode_action(action, observer),
        "action_canonicalized": canonical != action,
        "search_statistics": [],
        "outcome": {
            "team_points": team_points,
            "opponent_points": opponent_points,
            "point_diff": point_diff,
            "result": result,
            "won": result == "win",
        },
    }


def _prepare_game(job):
    path_string, validation_fraction, test_fraction, winners_only = job
    path = Path(path_string)
    source_game_id = int(path.stem)
    public_game_id = anonymized_game_id(source_game_id)
    split = split_for_game(
        source_game_id,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    game = parse_file(path)
    counts = Counter()
    records = []
    rounds_with_records = 0

    for round_index, round_ in enumerate(game.rounds):
        result = replay_round(round_, include_inferred_bomb_passes=True)
        counts["round_{}".format(result.status)] += 1
        if result.status != "ok":
            continue
        winner = winning_team(round_.scores)
        if winners_only and winner is None:
            counts["round_draw_skipped"] += 1
            continue
        written_in_round = 0
        for decision_index, (state, event, action) in enumerate(result.decisions):
            if winners_only and action.player_pos % 2 != winner:
                continue
            record = make_record(
                game_id=source_game_id,
                round_index=round_index,
                decision_index=decision_index,
                round_=round_,
                state=state,
                event=event,
                action=action,
            )
            if record["chosen_action"] not in record["legal_actions"]:
                raise AssertionError("chosen action missing from legal actions")
            records.append(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
            counts["{}_decisions".format(split)] += 1
            counts["{}_actions".format(record["decision_source"])] += 1
            counts["canonicalized_actions"] += int(record["action_canonicalized"])
            counts["result_{}_actions".format(record["outcome"]["result"])] += 1
            chosen = record["chosen_action"]
            action_label = chosen["type"]
            if chosen.get("combination"):
                action_label += "/{}".format(chosen["combination"])
            counts["action_{}".format(action_label)] += 1
            counts[
                "context_{}".format(record["observation"]["decision_context"])
            ] += 1
            if isinstance(action, PlayBomb):
                counts["played_bombs"] += 1
            written_in_round += 1
        if written_in_round:
            counts["{}_rounds".format(split)] += 1
            rounds_with_records += 1

    return {
        "game_id": public_game_id,
        "split": split,
        "records": records,
        "rounds_with_records": rounds_with_records,
        "counts": dict(counts),
    }


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_text_output(path, *, compress):
    if not compress:
        return Path(path).open("w", encoding="utf-8", newline="\n")
    raw = Path(path).open("wb")
    compressed = gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=raw,
        mtime=0,
    )
    return io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")


def _open_text_input(path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def validate_split_manifest(path):
    assignments = {}
    conflicts = set()
    rows = 0
    with _open_text_input(path) as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            previous = assignments.setdefault(row["game_id"], row["split"])
            if previous != row["split"]:
                conflicts.add(row["game_id"])
    return {
        "passed": not conflicts,
        "rows": rows,
        "unique_games": len(assignments),
        "game_ids_in_multiple_splits": len(conflicts),
    }


def _result_iterator(paths, *, validation_fraction, test_fraction, winners_only, workers):
    jobs = iter(
        (str(path), validation_fraction, test_fraction, winners_only)
        for path in paths
    )
    if workers == 1:
        yield from map(_prepare_game, jobs)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # ProcessPoolExecutor.map submits every path eagerly.  A full dataset
        # can contain millions of serialized records, so keep only a bounded
        # ordered queue of game results in memory.
        pending = []
        for _ in range(workers * 2):
            try:
                pending.append(executor.submit(_prepare_game, next(jobs)))
            except StopIteration:
                break
        while pending:
            future = pending.pop(0)
            yield future.result()
            try:
                pending.append(executor.submit(_prepare_game, next(jobs)))
            except StopIteration:
                pass


def prepare_dataset(
    *,
    input_dir,
    output_dir,
    validation_fraction=0.1,
    test_fraction=0.1,
    winners_only=False,
    workers=1,
    show_progress=False,
    max_games=None,
    compress=True,
):
    _validate_fractions(validation_fraction, test_fraction)
    if workers < 1:
        raise ValueError("workers must be positive")
    if max_games is not None and max_games < 1:
        raise ValueError("max_games must be positive")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(input_dir).glob("*.tch"), key=lambda path: int(path.stem))
    if max_games is not None:
        paths = paths[-max_games:]
    suffix = ".jsonl.gz" if compress else ".jsonl"
    temporary_paths = {
        split: output_dir / "{}{}.tmp".format(split, suffix) for split in SPLITS
    }
    manifest_temporary = output_dir / "split-manifest{}.tmp".format(suffix)
    for path in (*temporary_paths.values(), manifest_temporary):
        if path.exists():
            path.unlink()

    outputs = {
        split: _open_text_output(temporary_paths[split], compress=compress)
        for split in SPLITS
    }
    manifest = _open_text_output(manifest_temporary, compress=compress)
    counts = Counter()
    games_with_records = Counter()
    progress_step = max(1, len(paths) // 20)

    try:
        results = _result_iterator(
            paths,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            winners_only=winners_only,
            workers=workers,
        )
        for path_index, result in enumerate(results, start=1):
            split = result["split"]
            outputs[split].writelines(result["records"])
            counts.update(result["counts"])
            if result["records"]:
                games_with_records[split] += 1
            manifest.write(
                json.dumps(
                    {
                        "game_id": result["game_id"],
                        "split": split,
                        "rounds_with_records": result["rounds_with_records"],
                        "decisions": len(result["records"]),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            if show_progress and (
                path_index == len(paths) or path_index % progress_step == 0
            ):
                percent = 100.0 * path_index / max(1, len(paths))
                decisions = sum(
                    counts["{}_decisions".format(candidate)] for candidate in SPLITS
                )
                print(
                    "[dataset-v2] {}/{} files ({:.0f}%) decisions={}".format(
                        path_index, len(paths), percent, decisions
                    ),
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        for output in outputs.values():
            output.close()
        manifest.close()

    final_paths = {}
    for split, temporary_path in temporary_paths.items():
        final_path = output_dir / "{}{}".format(split, suffix)
        os.replace(temporary_path, final_path)
        final_paths[split] = final_path
    manifest_path = output_dir / "split-manifest{}".format(suffix)
    os.replace(manifest_temporary, manifest_path)
    leakage_check = validate_split_manifest(manifest_path)
    if not leakage_check["passed"]:
        raise AssertionError("a source game was assigned to multiple dataset splits")

    files = {
        split: {
            "path": final_paths[split].name,
            "bytes": final_paths[split].stat().st_size,
            "sha256": _sha256_file(final_paths[split]),
        }
        for split in SPLITS
    }
    files["split_manifest"] = {
        "path": manifest_path.name,
        "bytes": manifest_path.stat().st_size,
        "sha256": _sha256_file(manifest_path),
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "rules_version": RULES_VERSION,
        "source": "brettspielwelt",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split_namespace": SPLIT_NAMESPACE,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "winners_only": winners_only,
        "input_files": len(paths),
        "compression": "gzip" if compress else "none",
        "identity_policy": "player names omitted; source game ids pseudonymized",
        "chosen_action_membership_checked": True,
        "leakage_check": leakage_check,
        "games_with_records": {
            split: games_with_records[split] for split in SPLITS
        },
        "counts": dict(sorted(counts.items())),
        "files": files,
        "training_views": {
            "all": {"filter": None, "weight": "1.0"},
            "winners_only": {
                "filter": "outcome.result == 'win'",
                "weight": "1.0",
            },
            "result_weighted": {
                "filter": None,
                "weight": "selected during BC v2 validation from outcome.point_diff",
            },
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Build privacy-safe rules-v2 JSONL from verified BSW logs"
    )
    parser.add_argument("--input-dir", default="datasets/raw/brettspielwelt")
    parser.add_argument(
        "--output-dir", default="datasets/processed/brettspielwelt-decisions-v2"
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--winners-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--uncompressed", action="store_true")
    args = parser.parse_args()
    metadata = prepare_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        winners_only=args.winners_only,
        workers=args.workers,
        show_progress=True,
        max_games=args.max_games,
        compress=not args.uncompressed,
    )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
