import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from statistics import mean

from llm_client import chat_json
from prompts import render_aggregator, render_classifier_a, render_classifier_b, render_critic
from topics import normalize_topic_distribution, top_choice


def _json_dump(data):
    return json.dumps(data, ensure_ascii=False, indent=2)


def _confidence(value, default=0.5):
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _normalize_classifier_output(output):
    return {
        "topics": normalize_topic_distribution(output.get("topics", [])),
        "confidence": _confidence(output.get("confidence")),
        "rationale": str(output.get("rationale", "")),
    }


def _normalize_critic_output(output):
    return {
        "recommended_topics": normalize_topic_distribution(output.get("recommended_topics", [])),
        "confidence": _confidence(output.get("confidence")),
        "critic_notes": str(output.get("critic_notes", "")),
    }


def _normalize_aggregator_output(output):
    return {
        "final_topics": normalize_topic_distribution(output.get("final_topics", [])),
        "confidence": _confidence(output.get("confidence")),
        "rationale": str(output.get("rationale", "")),
    }


def detect_disagreement(classifier_a, classifier_b, percentage_threshold=0.25):
    topics_a = classifier_a["topics"]
    topics_b = classifier_b["topics"]
    if top_choice(topics_a) != top_choice(topics_b):
        return True

    percentages = {}
    for item in topics_a:
        percentages.setdefault(item["choice"], [0.0, 0.0])[0] = item["percentage"]
    for item in topics_b:
        percentages.setdefault(item["choice"], [0.0, 0.0])[1] = item["percentage"]

    return any(abs(a - b) >= percentage_threshold for a, b in percentages.values())


def average_topics(classifier_a, classifier_b):
    totals = {}
    for output in (classifier_a, classifier_b):
        for item in output["topics"]:
            totals[item["choice"]] = totals.get(item["choice"], 0.0) + item["percentage"] / 2
    return normalize_topic_distribution(
        [{"choice": choice, "percentage": percentage} for choice, percentage in totals.items()]
    )


def _classification_text(text, metadata):
    if not metadata:
        return text

    context_fields = [
        ("date", "Date"),
        ("house", "House"),
        ("nameOfHouse", "House"),
        ("meeting", "Meeting"),
        ("nameOfMeeting", "Meeting"),
        ("speaker", "Speaker"),
        ("speaker_group", "Speaker group"),
        ("speakerGroup", "Speaker group"),
    ]
    context = []
    seen_labels = set()
    for key, label in context_fields:
        value = metadata.get(key)
        if value in (None, "") or label in seen_labels:
            continue
        context.append(f"- {label}: {value}")
        seen_labels.add(label)

    if not context:
        return text
    return "Context:\n" + "\n".join(context) + "\n\nUtterance:\n" + text


def classify_text(
    text,
    speech_id=None,
    metadata=None,
    force_critic=False,
    disagreement_threshold=0.25,
):
    prompt_text = _classification_text(text, metadata)
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(chat_json, render_classifier_a(prompt_text), temperature=0.0)
        future_b = executor.submit(chat_json, render_classifier_b(prompt_text), temperature=0.0)
        classifier_a = _normalize_classifier_output(future_a.result())
        classifier_b = _normalize_classifier_output(future_b.result())

    disagreement = force_critic or detect_disagreement(
        classifier_a,
        classifier_b,
        percentage_threshold=disagreement_threshold,
    )
    critic = None

    if disagreement:
        critic = _normalize_critic_output(
            chat_json(render_critic(prompt_text, _json_dump(classifier_a), _json_dump(classifier_b)), temperature=0.0)
        )
        aggregated = _normalize_aggregator_output(
            chat_json(
                render_aggregator(
                    prompt_text,
                    _json_dump(classifier_a),
                    _json_dump(classifier_b),
                    _json_dump(critic),
                ),
                temperature=0.0,
            )
        )
    else:
        aggregated = {
            "final_topics": average_topics(classifier_a, classifier_b),
            "confidence": round(mean([classifier_a["confidence"], classifier_b["confidence"]]), 4),
            "rationale": "Classifier agents agreed, so their topic distributions were averaged.",
        }

    return {
        "speech_id": speech_id,
        "metadata": metadata or {},
        "text": text,
        "classifier_a": classifier_a,
        "classifier_b": classifier_b,
        "disagreement_detected": disagreement,
        "critic": critic,
        "final_topics": aggregated["final_topics"],
        "confidence": aggregated["confidence"],
        "rationale": aggregated["rationale"],
    }


def _load_records(input_path, text_field):
    _, ext = os.path.splitext(input_path.lower())
    if ext == ".jsonl":
        with open(input_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                yield str(record.get("speech_id") or record.get("id") or line_number), record[text_field], record
        return

    if ext == ".csv":
        with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for line_number, record in enumerate(reader, start=1):
                yield str(record.get("speech_id") or record.get("id") or line_number), record[text_field], record
        return

    raise ValueError("Input file must be .jsonl or .csv")


def main():
    parser = argparse.ArgumentParser(description="Multi-agent topic classifier for parliamentary utterances.")
    parser.add_argument("--text", help="Single utterance text to classify.")
    parser.add_argument("--input", help="CSV or JSONL file to classify.")
    parser.add_argument("--text_field", default="text", help="Field name containing utterance text.")
    parser.add_argument("--output", default="output/topic-classification-results.jsonl", help="Output JSONL path.")
    parser.add_argument("--force_critic", action="store_true", help="Always run critic and aggregator agents.")
    parser.add_argument(
        "--disagreement_threshold",
        type=float,
        default=0.25,
        help="Trigger critic when topic percentage difference is at least this value.",
    )
    args = parser.parse_args()

    if not args.text and not args.input:
        parser.error("Provide --text or --input.")
    if not 0.0 <= args.disagreement_threshold <= 1.0:
        parser.error("--disagreement_threshold must be between 0.0 and 1.0.")

    if args.text:
        result = classify_text(
            args.text,
            speech_id="single",
            force_critic=args.force_critic,
            disagreement_threshold=args.disagreement_threshold,
        )
        print(_json_dump(result))
        return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    count = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for speech_id, text, metadata in _load_records(args.input, args.text_field):
            result = classify_text(
                text=text,
                speech_id=speech_id,
                metadata=metadata,
                force_critic=args.force_critic,
                disagreement_threshold=args.disagreement_threshold,
            )
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            count += 1

    print(f"Saved {count} classification results to {args.output}")


if __name__ == "__main__":
    main()
