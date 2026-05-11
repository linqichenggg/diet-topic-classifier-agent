# diet-topic-classifier-agent

Multi-agent topic classifier for Japanese National Diet speech records.

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
