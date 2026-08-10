"""Convert engine-verified Brettspielwelt rounds to training JSONL."""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from gym_agents.training_data import encode_action, encode_observation
from scraper.brettspielwelt_parser import parse_file
from scraper.brettspielwelt_replay import canonical_action, replay_round


def split_for_game(game_id, validation_fraction):
    digest = hashlib.sha256("bsw:{}".format(game_id).encode("ascii")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64)
    return "validation" if bucket < validation_fraction else "train"


def winning_team(scores):
    if scores[0] == scores[1]:
        return None
    return 0 if scores[0] > scores[1] else 1


def make_record(*, game_id, round_index, decision_index, round_, state, action):
    observer = action.player_pos
    canonical = canonical_action(state, action)
    if canonical is None:
        raise ValueError("human action has no canonical legal action")
    team_points = round_.scores[observer % 2]
    opponent_points = round_.scores[(observer + 1) % 2]
    return {
        "schema_version": 1,
        "source": "brettspielwelt",
        "game_id": "bsw-{}".format(game_id),
        "round": round_index,
        "decision": decision_index,
        "player": observer,
        "agent": "human",
        "observation": encode_observation(state, observer),
        "legal_actions": [
            encode_action(legal_action, observer)
            for legal_action in sorted(state.possible_actions_list, key=repr)
        ],
        "chosen_action": encode_action(canonical, observer),
        "original_chosen_action": encode_action(action, observer),
        "action_canonicalized": canonical != action,
        "search_statistics": [],
        "outcome": {
            "team_points": team_points,
            "opponent_points": opponent_points,
            "point_diff": team_points - opponent_points,
            "won": team_points > opponent_points,
        },
    }


def prepare_dataset(
    *,
    input_dir,
    output_dir,
    validation_fraction=0.1,
    winners_only=True,
):
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        split: (output_dir / "{}.jsonl".format(split)).open(
            "w", encoding="utf-8", newline="\n"
        )
        for split in ("train", "validation")
    }
    counts = Counter()
    game_ids = set()
    try:
        for path in sorted(Path(input_dir).glob("*.tch")):
            game_id = int(path.stem)
            game = parse_file(path)
            split = split_for_game(game_id, validation_fraction)
            for round_index, round_ in enumerate(game.rounds):
                result = replay_round(round_)
                counts["round_{}".format(result.status)] += 1
                if result.status != "ok":
                    continue
                winner = winning_team(round_.scores)
                if winners_only and winner is None:
                    counts["round_draw_skipped"] += 1
                    continue
                written_in_round = 0
                for decision_index, (state, _, action) in enumerate(result.decisions):
                    if winners_only and action.player_pos % 2 != winner:
                        continue
                    record = make_record(
                        game_id=game_id,
                        round_index=round_index,
                        decision_index=decision_index,
                        round_=round_,
                        state=state,
                        action=action,
                    )
                    if record["chosen_action"] not in record["legal_actions"]:
                        raise AssertionError("chosen action missing from legal actions")
                    outputs[split].write(
                        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    counts["{}_decisions".format(split)] += 1
                    counts["canonicalized_actions"] += int(
                        record["action_canonicalized"]
                    )
                    written_in_round += 1
                if written_in_round:
                    counts["{}_rounds".format(split)] += 1
                    game_ids.add(game_id)
    finally:
        for output in outputs.values():
            output.close()

    metadata = {
        "schema_version": 1,
        "source": "brettspielwelt",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "validation_fraction": validation_fraction,
        "winners_only": winners_only,
        "games_with_records": len(game_ids),
        "counts": dict(sorted(counts.items())),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Build privacy-safe training JSONL from verified BSW logs"
    )
    parser.add_argument("--input-dir", default="datasets/raw/brettspielwelt")
    parser.add_argument(
        "--output-dir", default="datasets/processed/brettspielwelt-decisions-v1"
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--all-teams", action="store_true")
    args = parser.parse_args()
    metadata = prepare_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        validation_fraction=args.validation_fraction,
        winners_only=not args.all_teams,
    )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
