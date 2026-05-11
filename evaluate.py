import argparse
import csv
import json
import os

from classifier import classify_text
from topics import TOPIC_BY_CHOICE, top_choice


def _load_labeled_records(input_path):
    _, ext = os.path.splitext(input_path.lower())
    if ext == ".jsonl":
        with open(input_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                record["_line_number"] = line_number
                yield record
        return

    if ext == ".csv":
        with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for line_number, record in enumerate(reader, start=2):
                record["_line_number"] = line_number
                yield record
        return

    raise ValueError("Input file must be .jsonl or .csv")


def _parse_gold_choice(value):
    try:
        choice = int(value)
    except (TypeError, ValueError):
        return None
    if choice not in TOPIC_BY_CHOICE:
        return None
    return choice


def _safe_accuracy(correct, total):
    if total == 0:
        return None
    return round(correct / total, 4)


def _empty_topic_stats():
    return {
        "total": 0,
        "final_correct": 0,
        "classifier_a_correct": 0,
        "classifier_b_correct": 0,
    }


def _evaluate_result(result, gold_choice):
    final_choice = top_choice(result["final_topics"])
    classifier_a_choice = top_choice(result["classifier_a"]["topics"])
    classifier_b_choice = top_choice(result["classifier_b"]["topics"])
    return {
        "final_choice": final_choice,
        "classifier_a_choice": classifier_a_choice,
        "classifier_b_choice": classifier_b_choice,
        "final_correct": final_choice == gold_choice,
        "classifier_a_correct": classifier_a_choice == gold_choice,
        "classifier_b_correct": classifier_b_choice == gold_choice,
    }


def evaluate(args):
    total = 0
    skipped = 0
    errors = 0
    final_correct = 0
    classifier_a_correct = 0
    classifier_b_correct = 0
    disagreement_count = 0
    critic_count = 0
    by_gold_topic = {}

    predictions_output = None
    if args.predictions_output:
        os.makedirs(os.path.dirname(args.predictions_output) or ".", exist_ok=True)
        predictions_output = open(args.predictions_output, "w", encoding="utf-8")

    try:
        for record in _load_labeled_records(args.input):
            gold_choice = _parse_gold_choice(record.get(args.gold_field))
            text = record.get(args.text_field)
            if gold_choice is None or not text:
                skipped += 1
                continue

            speech_id = str(record.get("speech_id") or record.get("id") or record.get("_line_number"))
            print(f"Evaluating {speech_id}...", flush=True)
            try:
                result = classify_text(
                    text=str(text),
                    speech_id=speech_id,
                    metadata=record,
                    force_critic=args.force_critic,
                    disagreement_threshold=args.disagreement_threshold,
                )
            except Exception as e:
                errors += 1
                if predictions_output:
                    predictions_output.write(
                        json.dumps(
                            {
                                "speech_id": speech_id,
                                "gold_choice": gold_choice,
                                "error": str(e),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    predictions_output.flush()
                print(f"Error on {speech_id}: {e}", flush=True)
                continue

            item_eval = _evaluate_result(result, gold_choice)

            total += 1
            final_correct += int(item_eval["final_correct"])
            classifier_a_correct += int(item_eval["classifier_a_correct"])
            classifier_b_correct += int(item_eval["classifier_b_correct"])
            disagreement_count += int(bool(result["disagreement_detected"]))
            critic_count += int(result["critic"] is not None)

            topic_key = str(gold_choice)
            topic_stats = by_gold_topic.setdefault(topic_key, _empty_topic_stats())
            topic_stats["total"] += 1
            topic_stats["final_correct"] += int(item_eval["final_correct"])
            topic_stats["classifier_a_correct"] += int(item_eval["classifier_a_correct"])
            topic_stats["classifier_b_correct"] += int(item_eval["classifier_b_correct"])

            if predictions_output:
                predictions_output.write(
                    json.dumps(
                        {
                            "speech_id": speech_id,
                            "gold_choice": gold_choice,
                            **item_eval,
                            "disagreement_detected": result["disagreement_detected"],
                            "critic_used": result["critic"] is not None,
                            "result": result,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                predictions_output.flush()
    finally:
        if predictions_output:
            predictions_output.close()

    for choice, stats in by_gold_topic.items():
        stats["label"] = TOPIC_BY_CHOICE[int(choice)]["label"]
        stats["ja_label"] = TOPIC_BY_CHOICE[int(choice)]["ja_label"]
        stats["final_top1_accuracy"] = _safe_accuracy(stats["final_correct"], stats["total"])
        stats["classifier_a_top1_accuracy"] = _safe_accuracy(stats["classifier_a_correct"], stats["total"])
        stats["classifier_b_top1_accuracy"] = _safe_accuracy(stats["classifier_b_correct"], stats["total"])

    summary = {
        "input": args.input,
        "total": total,
        "skipped": skipped,
        "errors": errors,
        "final_top1_accuracy": _safe_accuracy(final_correct, total),
        "classifier_a_top1_accuracy": _safe_accuracy(classifier_a_correct, total),
        "classifier_b_top1_accuracy": _safe_accuracy(classifier_b_correct, total),
        "disagreement_rate": _safe_accuracy(disagreement_count, total),
        "critic_rate": _safe_accuracy(critic_count, total),
        "disagreement_threshold": args.disagreement_threshold,
        "force_critic": args.force_critic,
        "by_gold_topic": by_gold_topic,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved evaluation summary to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate topic classifier outputs against human labels.")
    parser.add_argument("--input", required=True, help="Human-labeled CSV or JSONL file.")
    parser.add_argument("--text_field", default="speech", help="Field name containing speech text.")
    parser.add_argument("--gold_field", default="gold_choice", help="Field name containing the human topic label.")
    parser.add_argument("--output", default="output/topic-classifier-eval.json", help="Evaluation summary JSON path.")
    parser.add_argument("--predictions_output", help="Optional JSONL path for per-record predictions.")
    parser.add_argument("--force_critic", action="store_true", help="Always run critic and aggregator agents.")
    parser.add_argument(
        "--disagreement_threshold",
        type=float,
        default=0.25,
        help="Trigger critic when topic percentage difference is at least this value.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.disagreement_threshold <= 1.0:
        parser.error("--disagreement_threshold must be between 0.0 and 1.0.")

    evaluate(args)


if __name__ == "__main__":
    main()
