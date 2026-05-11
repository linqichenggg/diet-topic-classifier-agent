import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone


API_ENDPOINT = "https://kokkai.ndl.go.jp/api/speech"
MAX_PAGE_SIZE = 100


def _manifest_path(output_path):
    root, ext = os.path.splitext(output_path)
    if ext == ".jsonl":
        return f"{root}.manifest.json"
    return f"{output_path}.manifest.json"


def _count_jsonl_records(path):
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _build_base_params(args):
    params = {
        "recordPacking": "json",
    }
    if args.session_from is not None:
        params["sessionFrom"] = str(args.session_from)
    if args.session_to is not None:
        params["sessionTo"] = str(args.session_to)
    if args.from_date:
        params["from"] = args.from_date
    if args.until_date:
        params["until"] = args.until_date
    if args.house:
        params["nameOfHouse"] = args.house
    if args.meeting:
        params["nameOfMeeting"] = args.meeting
    if args.speaker:
        params["speaker"] = args.speaker
    if args.any:
        params["any"] = args.any
    return params


def _fetch_page(params, timeout):
    url = f"{API_ENDPOINT}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return url, json.loads(body)


def _as_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_record(record):
    return {
        "speech_id": record.get("speechID"),
        "session": record.get("session"),
        "house": record.get("nameOfHouse"),
        "meeting": record.get("nameOfMeeting"),
        "issue": record.get("issue"),
        "issue_id": record.get("issueID"),
        "date": record.get("date"),
        "speaker": record.get("speaker"),
        "speaker_yomi": record.get("speakerYomi"),
        "speaker_group": record.get("speakerGroup"),
        "speaker_position": record.get("speakerPosition"),
        "speaker_role": record.get("speakerRole"),
        "speech_order": record.get("speechOrder"),
        "speech": record.get("speech"),
        "speech_url": record.get("speechURL"),
        "meeting_url": record.get("meetingURL"),
        "pdf_url": record.get("pdfURL"),
        "start_page": record.get("startPage"),
        "closing": record.get("closing"),
    }


def fetch_speeches(args):
    base_params = _build_base_params(args)
    if not any(
        key in base_params
        for key in ("sessionFrom", "sessionTo", "from", "until", "nameOfHouse", "nameOfMeeting", "speaker", "any")
    ):
        raise ValueError("Provide at least one search condition, such as --session_from or --from_date.")

    page_size = max(1, min(MAX_PAGE_SIZE, args.page_size))
    existing_records = _count_jsonl_records(args.output) if args.resume else 0
    start_record = existing_records + 1
    records_written = 0
    api_reported_total = None
    first_request_url = None
    last_request_url = None
    started_at = datetime.now(timezone.utc).isoformat()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    mode = "a" if args.resume and existing_records else "w"

    with open(args.output, mode, encoding="utf-8") as output:
        while True:
            if args.limit is not None:
                remaining = args.limit - existing_records - records_written
                if remaining <= 0:
                    break
                current_page_size = min(page_size, remaining)
            else:
                current_page_size = page_size

            params = dict(base_params)
            params["startRecord"] = str(start_record)
            params["maximumRecords"] = str(current_page_size)

            url, data = _fetch_page(params, timeout=args.timeout)
            first_request_url = first_request_url or url
            last_request_url = url
            api_reported_total = _as_int(data.get("numberOfRecords"), api_reported_total or 0)

            records = data.get("speechRecord") or []
            if isinstance(records, dict):
                records = [records]
            if not records:
                break

            for record in records:
                output.write(json.dumps(_normalize_record(record), ensure_ascii=False) + "\n")
                records_written += 1
            output.flush()

            start_record += len(records)
            print(f"Fetched {existing_records + records_written} records", flush=True)

            if start_record > api_reported_total:
                break
            if len(records) < current_page_size:
                break
            if args.limit is not None and existing_records + records_written >= args.limit:
                break
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    manifest = {
        "source": API_ENDPOINT,
        "query": base_params,
        "output": args.output,
        "resume": args.resume,
        "existing_records": existing_records,
        "records_written_this_run": records_written,
        "records_written_total": existing_records + records_written,
        "api_reported_total": api_reported_total,
        "limit": args.limit,
        "page_size": page_size,
        "sleep_seconds": args.sleep_seconds,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "first_request_url": first_request_url,
        "last_request_url": last_request_url,
    }
    manifest_path = _manifest_path(args.output)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Saved {existing_records + records_written} records to {args.output}")
    print(f"Saved manifest to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch speech records from the National Diet API.")
    parser.add_argument("--session_from", type=int, help="First Diet session number, for example 198.")
    parser.add_argument("--session_to", type=int, help="Last Diet session number, for example 216.")
    parser.add_argument("--from_date", help="Start meeting date in YYYY-MM-DD format.")
    parser.add_argument("--until_date", help="End meeting date in YYYY-MM-DD format.")
    parser.add_argument("--house", help="House name, for example 衆議院 or 参議院.")
    parser.add_argument("--meeting", help="Meeting name filter.")
    parser.add_argument("--speaker", help="Speaker name filter.")
    parser.add_argument("--any", help="Speech text keyword filter.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--limit", type=int, help="Maximum total records to keep in the output file.")
    parser.add_argument("--page_size", type=int, default=MAX_PAGE_SIZE, help="Records per API request. Max is 100.")
    parser.add_argument("--sleep_seconds", type=float, default=3.0, help="Delay between API requests.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP request timeout in seconds.")
    parser.add_argument("--resume", action="store_true", help="Append after existing JSONL records.")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1.")
    if args.page_size < 1:
        parser.error("--page_size must be at least 1.")
    if args.sleep_seconds < 0:
        parser.error("--sleep_seconds must be non-negative.")

    fetch_speeches(args)


if __name__ == "__main__":
    main()
