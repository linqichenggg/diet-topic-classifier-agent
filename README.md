# diet-topic-classifier-agent

Multi-agent topic classifier for Japanese National Diet speech records.

Here, "Diet" means the National Diet of Japan (国会), not food or nutrition.

## What It Does

The classifier uses four role-based LLM agents:

- `Classifier Agent A`: classifies by explicit policy domain.
- `Classifier Agent B`: classifies by speaker intent and parliamentary context.
- `Critic Agent`: reviews disagreements between the two classifiers.
- `Aggregator Agent`: produces the final weighted topic distribution.

`Classifier Agent A` and `Classifier Agent B` run in parallel. The critic is triggered only when the two classifier outputs disagree or when `--force_critic` is set.

## Files

- `topics.py`: 19-topic schema and output normalization.
- `prompts.py`: prompts for the four agents.
- `llm_client.py`: DeepSeek chat API client and JSON parsing.
- `classifier.py`: CLI and multi-agent classification pipeline.
- `fetch_kokkai_speeches.py`: fetch speech records from the official National Diet API.
- `evaluate.py`: evaluate model predictions against human-labeled `gold_choice` data.

## Data Source

Speech records can be fetched from the official National Diet Library API:

- https://kokkai.ndl.go.jp/api.html
- `https://kokkai.ndl.go.jp/api/speech`

The fetch script writes JSONL records and a manifest file recording query parameters, fetch time, and API-reported totals.

## Configuration

Set DeepSeek credentials with environment variables:

```bash
export DEEPSEEK_API_KEY="your-api-key"
export DEEPSEEK_MODEL="deepseek-chat"
```

Or create a local `secrets.local.json` file in this folder:

```json
{
  "deepseek_api_key": "your-api-key",
  "deepseek_model": "deepseek-chat",
  "deepseek_base_url": "https://api.deepseek.com/chat/completions"
}
```

`secrets.local.json` is ignored by git.

## Fetch Sample Speeches

```bash
python3 -u fetch_kokkai_speeches.py \
  --session_from 216 \
  --session_to 216 \
  --limit 3 \
  --sleep_seconds 1 \
  --output kokkai_speeches_sample.jsonl
```

## Classify One Utterance

```bash
python3 -u classifier.py \
  --text "再生可能エネルギーの導入拡大と電力の安定供給について政府の見解を伺います。"
```

## Classify JSONL/CSV

```bash
python3 -u classifier.py \
  --input kokkai_speeches_sample.jsonl \
  --text_field speech \
  --output output/topic-classification-results.jsonl \
  --disagreement_threshold 0.25
```

## Evaluate With Human Labels

The input CSV/JSONL should include:

- `speech`: utterance text
- `gold_choice`: human-labeled topic number from `topics.py`

Example:

```csv
speech_id,speech,gold_choice
sample-1,再生可能エネルギーの導入拡大について政府の見解を伺います。,13
```

Run evaluation:

```bash
python3 -u evaluate.py \
  --input topic_classifier_eval_sample.csv \
  --text_field speech \
  --gold_field gold_choice \
  --output output/topic-classifier-eval.json \
  --predictions_output output/topic-classifier-eval-predictions.jsonl
```

## Pilot Result

In a 50-sample human-labeled pilot set, the latest version matched 49 out of 50 human labels.

This is only a small pilot result. It should not be treated as a final accuracy claim until the evaluation set covers more sessions, speakers, and topic categories.
