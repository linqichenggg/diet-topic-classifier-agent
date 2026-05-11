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

