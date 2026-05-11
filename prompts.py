from topics import format_topic_list


CLASSIFIER_A_PROMPT = """You are Classifier Agent A.
Your task is to classify one Japanese parliamentary utterance into the fixed policy topics below.
Focus on the policy domain explicitly discussed in the utterance.

Policy topic list:
{topic_list}

Return JSON only:
{{
  "topics": [
    {{"choice": 13, "percentage": 0.7}},
    {{"choice": 2, "percentage": 0.3}}
  ],
  "confidence": 0.0,
  "rationale": "short explanation"
}}

Rules:
- Percentages must sum to 1.0.
- Use multiple topics only when the utterance substantively covers multiple policy domains.
- Use choice 18 only when the content cannot be classified.
- Use choice 19 only for purely procedural or non-policy utterances with no identifiable policy domain.
- Do not choose choice 19 just because the utterance is short, technical, an answer from an official, or about implementation details.
- If the utterance refers to a specific policy object, law, ministry action, subsidy, regulation, or international issue, classify by that underlying policy domain.
- North Korea, abductees, missiles, defense posture, alliances, and international order belong to choice 11.
- Political funds, party branch reporting, election administration, public-sector digital systems, and government reform belong to choice 3.
- Disaster prevention, earthquake reconstruction, and emergency response belong to choice 14.
- Food production rules affecting farmers, pickled products, genome-edited food, and local agricultural business belong to choice 17.

Utterance:
{text}
"""


CLASSIFIER_B_PROMPT = """You are Classifier Agent B.
Classify the Japanese parliamentary utterance into the fixed policy topics below.
Focus on the speaker's intent, policy implications, and parliamentary context.

Policy topic list:
{topic_list}

Return JSON only:
{{
  "topics": [
    {{"choice": 7, "percentage": 0.6}},
    {{"choice": 2, "percentage": 0.4}}
  ],
  "confidence": 0.0,
  "rationale": "short explanation"
}}

Rules:
- Percentages must sum to 1.0.
- Prefer a single main topic when the utterance has a clear center.
- Add secondary topics only when they are necessary.
- Use choice 18 for unclear content.
- Use choice 19 only for purely procedural or non-policy utterances with no identifiable policy domain.
- Do not choose choice 19 just because the utterance is short, technical, an answer from an official, or about implementation details.
- If the utterance refers to a specific policy object, law, ministry action, subsidy, regulation, or international issue, classify by that underlying policy domain.
- North Korea, abductees, missiles, defense posture, alliances, and international order belong to choice 11.
- Political funds, party branch reporting, election administration, public-sector digital systems, and government reform belong to choice 3.
- Disaster prevention, earthquake reconstruction, and emergency response belong to choice 14.
- Food production rules affecting farmers, pickled products, genome-edited food, and local agricultural business belong to choice 17.

Utterance:
{text}
"""


CRITIC_PROMPT = """You are a Critic Agent checking topic classification disagreements.
Review the utterance and the two classifier outputs. Identify which classification is more plausible, or propose a better distribution.

Policy topic list:
{topic_list}

Utterance:
{text}

Classifier A:
{classifier_a}

Classifier B:
{classifier_b}

Return JSON only:
{{
  "critic_notes": "explain the disagreement and the strongest interpretation",
  "recommended_topics": [
    {{"choice": 13, "percentage": 1.0}}
  ],
  "confidence": 0.0
}}
"""


AGGREGATOR_PROMPT = """You are an Aggregator Agent.
Produce the final topic distribution for the utterance by considering Classifier A, Classifier B, and the Critic Agent.

Policy topic list:
{topic_list}

Utterance:
{text}

Classifier A:
{classifier_a}

Classifier B:
{classifier_b}

Critic:
{critic}

Return JSON only:
{{
  "final_topics": [
    {{"choice": 13, "percentage": 1.0}}
  ],
  "confidence": 0.0,
  "rationale": "short explanation of the final decision"
}}
"""


def render_classifier_a(text):
    return CLASSIFIER_A_PROMPT.format(topic_list=format_topic_list(), text=text)


def render_classifier_b(text):
    return CLASSIFIER_B_PROMPT.format(topic_list=format_topic_list(), text=text)


def render_critic(text, classifier_a, classifier_b):
    return CRITIC_PROMPT.format(
        topic_list=format_topic_list(),
        text=text,
        classifier_a=classifier_a,
        classifier_b=classifier_b,
    )


def render_aggregator(text, classifier_a, classifier_b, critic):
    return AGGREGATOR_PROMPT.format(
        topic_list=format_topic_list(),
        text=text,
        classifier_a=classifier_a,
        classifier_b=classifier_b,
        critic=critic,
    )
