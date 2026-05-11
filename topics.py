TOPICS = [
    {"choice": 1, "label": "Economy", "ja_label": "経済"},
    {"choice": 2, "label": "Fiscal Policy and Tax", "ja_label": "財政・税"},
    {"choice": 3, "label": "Administrative and Political Reform", "ja_label": "行政・政治改革"},
    {"choice": 4, "label": "Work Style", "ja_label": "働き方"},
    {"choice": 5, "label": "Childcare and Education", "ja_label": "子育て・教育"},
    {"choice": 6, "label": "Youth and Higher Education", "ja_label": "若者・高等教育"},
    {"choice": 7, "label": "Pensions", "ja_label": "年金"},
    {"choice": 8, "label": "Medical Care and Nursing Care", "ja_label": "医療・介護"},
    {"choice": 9, "label": "Infectious Disease Measures", "ja_label": "感染症対策"},
    {"choice": 10, "label": "Japan-US Relations", "ja_label": "日米関係"},
    {"choice": 11, "label": "Diplomacy and Security", "ja_label": "外交・安全保障"},
    {"choice": 12, "label": "Constitutional Revision and Imperial Household", "ja_label": "憲法改正・皇室"},
    {"choice": 13, "label": "Energy and Environment", "ja_label": "エネルギー・環境"},
    {"choice": 14, "label": "Reconstruction and Disasters", "ja_label": "復興・災害"},
    {"choice": 15, "label": "Gender and LGBTQ", "ja_label": "ジェンダー・LGBTQ"},
    {"choice": 16, "label": "Immigration and Foreign Residents", "ja_label": "移民・外国人"},
    {"choice": 17, "label": "Agriculture and Regional Revitalization", "ja_label": "農政・地方創生"},
    {"choice": 18, "label": "Unknown", "ja_label": "わからない"},
    {"choice": 19, "label": "Other", "ja_label": "その他"},
]

TOPIC_BY_CHOICE = {topic["choice"]: topic for topic in TOPICS}


def format_topic_list():
    return "\n".join(
        f"{topic['choice']}. {topic['ja_label']} ({topic['label']})" for topic in TOPICS
    )


def normalize_topic_distribution(topics):
    cleaned = []
    total = 0.0
    for item in topics or []:
        try:
            choice = int(item.get("choice"))
            percentage = float(item.get("percentage"))
        except (TypeError, ValueError, AttributeError):
            continue
        if choice not in TOPIC_BY_CHOICE or percentage <= 0:
            continue
        cleaned.append(
            {
                "choice": choice,
                "label": TOPIC_BY_CHOICE[choice]["label"],
                "ja_label": TOPIC_BY_CHOICE[choice]["ja_label"],
                "percentage": percentage,
            }
        )
        total += percentage

    if not cleaned:
        return [{"choice": 18, "label": "Unknown", "ja_label": "わからない", "percentage": 1.0}]

    if total <= 0:
        return [{"choice": 18, "label": "Unknown", "ja_label": "わからない", "percentage": 1.0}]

    normalized = []
    for item in cleaned:
        normalized.append({**item, "percentage": round(item["percentage"] / total, 4)})
    return sorted(normalized, key=lambda item: item["percentage"], reverse=True)


def top_choice(topics):
    normalized = normalize_topic_distribution(topics)
    return normalized[0]["choice"]

