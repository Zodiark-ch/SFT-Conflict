"""Decoder-only prompt and verbalizer contracts for the eight GLUE tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .registry import RenderedExample


@dataclass(frozen=True)
class GlueTaskSpec:
    key: str
    fields: tuple[str, ...]
    templates: tuple[str, ...]
    label_texts: tuple[str, ...]
    metric_names: tuple[str, ...]
    regression: bool = False


GLUE_TASK_SPECS: dict[str, GlueTaskSpec] = {
    "cola": GlueTaskSpec(
        key="cola",
        fields=("sentence",),
        templates=(
            'Sentence: "{sentence}"\nIs this sentence grammatically acceptable English? '
            "Answer acceptable or unacceptable.\nAnswer:",
            'Consider the English sentence: "{sentence}"\nIs its grammar acceptable? '
            "Answer acceptable or unacceptable.\nAnswer:",
        ),
        label_texts=("unacceptable", "acceptable"),
        metric_names=("matthews_correlation",),
    ),
    "sst2": GlueTaskSpec(
        key="sst2",
        fields=("sentence",),
        templates=(
            'Sentence: "{sentence}"\nWhat is its sentiment? Answer positive or negative.\n'
            "Sentiment:",
            'Review: "{sentence}"\nClassify the review as positive or negative.\nSentiment:',
        ),
        label_texts=("negative", "positive"),
        metric_names=("accuracy",),
    ),
    "mrpc": GlueTaskSpec(
        key="mrpc",
        fields=("sentence1", "sentence2"),
        templates=(
            'Sentence 1: "{sentence1}"\nSentence 2: "{sentence2}"\nDo these sentences '
            "have the same meaning? Answer yes or no.\nAnswer:",
            'First sentence: "{sentence1}"\nSecond sentence: "{sentence2}"\nAre they '
            "paraphrases? Answer yes or no.\nAnswer:",
        ),
        label_texts=("no", "yes"),
        metric_names=("accuracy", "f1", "averaged_scores"),
    ),
    "stsb": GlueTaskSpec(
        key="stsb",
        fields=("sentence1", "sentence2"),
        templates=(
            'Sentence 1: "{sentence1}"\nSentence 2: "{sentence2}"\nRate their semantic '
            "similarity from 0.0 (unrelated) to 5.0 (equivalent). Output one number.\nScore:",
            'Compare: "{sentence1}"\nWith: "{sentence2}"\nGive a similarity score between '
            "0.0 and 5.0. Output one number.\nScore:",
        ),
        label_texts=(),
        metric_names=("pearson", "spearmanr", "averaged_scores"),
        regression=True,
    ),
    "qqp": GlueTaskSpec(
        key="qqp",
        fields=("question1", "question2"),
        templates=(
            'Question 1: "{question1}"\nQuestion 2: "{question2}"\nAre these duplicate '
            "questions with the same intent? Answer yes or no.\nAnswer:",
            'First question: "{question1}"\nSecond question: "{question2}"\nDo they ask '
            "the same thing? Answer yes or no.\nAnswer:",
        ),
        label_texts=("no", "yes"),
        metric_names=("accuracy", "f1", "averaged_scores"),
    ),
    "mnli": GlueTaskSpec(
        key="mnli",
        fields=("premise", "hypothesis"),
        templates=(
            'Premise: "{premise}"\nHypothesis: "{hypothesis}"\nClassify the relationship as '
            "entailment, neutral, or contradiction.\nAnswer:",
            'Given the premise: "{premise}"\nDoes the hypothesis "{hypothesis}" follow, '
            "conflict, or remain unknown? Answer entailment, contradiction, or neutral.\nAnswer:",
        ),
        label_texts=("entailment", "neutral", "contradiction"),
        metric_names=("accuracy",),
    ),
    "qnli": GlueTaskSpec(
        key="qnli",
        fields=("question", "sentence"),
        templates=(
            'Question: "{question}"\nSentence: "{sentence}"\nDoes the sentence contain an '
            "answer to the question? Answer yes or no.\nAnswer:",
            'Given the question "{question}", can it be answered by the sentence '
            '"{sentence}"? Answer yes or no.\nAnswer:',
        ),
        label_texts=("yes", "no"),
        metric_names=("accuracy",),
    ),
    "rte": GlueTaskSpec(
        key="rte",
        fields=("sentence1", "sentence2"),
        templates=(
            'Given the premise: "{sentence1}"\nDoes this imply that "{sentence2}"? '
            "Please answer yes or no.\nAnswer:",
            'Premise: "{sentence1}"\nHypothesis: "{sentence2}"\nIs the hypothesis entailed? '
            "Answer yes or no.\nAnswer:",
        ),
        label_texts=("yes", "no"),
        metric_names=("accuracy",),
    ),
}


GLUE_REFERENCE_LEARNING_RATES: dict[str, dict[str, float]] = {
    "bert-base-uncased": {
        "cola": 5e-5,
        "sst2": 1e-5,
        "mrpc": 5e-5,
        "stsb": 5e-5,
        "qqp": 1e-5,
        "mnli": 1e-5,
        "qnli": 1e-5,
        "rte": 1e-5,
    },
    "roberta-base": {
        "cola": 1e-5,
        "sst2": 1e-5,
        "mrpc": 5e-5,
        "stsb": 1e-5,
        "qqp": 1e-5,
        "mnli": 1e-5,
        "qnli": 1e-5,
        "rte": 1e-5,
    },
}


GLUE_DECODER_LR_SEARCH_GRIDS: dict[str, tuple[float, ...]] = {
    task_name: (
        (1e-5, 2e-5, 5e-5)
        if 5e-5 in {rates[task_name] for rates in GLUE_REFERENCE_LEARNING_RATES.values()}
        else (5e-6, 1e-5, 2e-5)
    )
    for task_name in GLUE_TASK_SPECS
}


def render_glue(row: dict[str, Any], *, task_name: str) -> RenderedExample:
    spec = GLUE_TASK_SPECS[task_name]
    source_index = int(row.get("idx", 0))
    template_index = source_index % len(spec.templates)
    values = {field: str(row[field]) for field in spec.fields}
    prompt = spec.templates[template_index].format(**values)
    gold = float(row["label"]) if spec.regression else int(row["label"])
    label_text = f"{gold:.1f}" if spec.regression else spec.label_texts[int(gold)]
    return RenderedExample(
        prompt=prompt,
        response=f" {label_text}",
        metadata={
            "gold_label": gold,
            "label_text": label_text,
            "label_candidates": list(spec.label_texts),
            "metric_names": list(spec.metric_names),
            "task_name": task_name,
            "task_type": "regression" if spec.regression else "classification",
            "template_id": f"{task_name}-{template_index}",
        },
    )
