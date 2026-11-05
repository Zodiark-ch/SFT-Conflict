# Unifying and Beyond the Theory of Pervasive Capability Degradation in LLM Post-training SFT from Feature Interference

Code for the paper above ([arXiv:2611.05003](https://arxiv.org/abs/2611.05003)). See
[Citation](#citation) for how to cite it.

## Introduction

This work uses superposition and the geometry of the Laguerre–Voronoi diagram (LVD) to explain
why supervised fine-tuning (SFT) inevitably leads to pervasive capability degradation (PCD). It
further unifies existing methods into six paradigms (Section 3 of the paper). Building on this
analysis, we propose AEGIS, a general, scalable and interpretable loss function.

**Motivation.** Our analysis of cross-entropy shows that it cannot separate feature interference
from task adaptation, and that its one-hot targets require a massive number of samples to keep
features sparse. AEGIS instead supervises how far activations drift toward each token direction.
It therefore focuses on feature interference, and because it supervises many token directions at
once, it greatly reduces the number of samples needed.

In the code, AEGIS is `--method target_plus_anchor_ce_geometry` and the pervasive dataset is
called the *Anchor* dataset.

## Environment and setup

Experiments ran on 4 × NVIDIA A100 with Python 3.10, PyTorch 2.5.1 (CUDA 12.1),
Transformers 4.48.3, Accelerate 1.3.0, Datasets 3.2.0 and PEFT 0.14.0. The pervasive-capability
suite uses lm-evaluation-harness 0.4.9 in a separate environment. Exact pins are in
[`requirements/`](requirements).

Models: `mistral_7b` (Mistral-7B-v0.3), `qwen25_7b`, `qwen25_3b` and `qwen25_0.5b` (Qwen2.5),
each at a pinned revision.

Default training setup:

| Setting | Value |
|---|---|
| Fine-tuning | full fine-tuning, FP32, FSDP over 4 GPUs |
| Optimizer | AdamW, lr 1e-5, weight decay 0.01, linear schedule, 6% warm-up |
| Epochs / length | 3 epochs, 512 tokens, loss on responses only |
| Batch | 1 target + 1 pervasive sequence per GPU, 4 accumulation steps |
| Pervasive CE weight | 0.2 |
| Seed | 42 |
| Checkpoint selection | native validation accuracy for GLUE classification tasks, validation loss otherwise |

GLUE tasks are trained on 90% of the official training set, validated on the other 10%, and
tested on the official validation set. Mistral-7B uses about 37 GB per GPU; AEGIS on RTE takes
about 3 hours.

## Installation

```bash
git clone https://github.com/Zodiark-ch/SFT-Conflict.git
cd SFT-Conflict

# Training environment. When it lives in <artifact-root>/envs/train, run.py switches to it
# automatically; any other activated environment works as well.
python3.10 -m venv ../artifacts/envs/train
../artifacts/envs/train/bin/pip install -r requirements/train.lock \
    --extra-index-url https://download.pytorch.org/whl/cu121
../artifacts/envs/train/bin/pip install -e . --no-deps

# Mistral-7B-v0.3 and Infinity-Instruct are gated on Hugging Face.
huggingface-cli login

# Optional: pervasive-capability suite (lm-evaluation-harness 0.4.9).
python3.10 -m venv ../artifacts/envs/lm-eval
../artifacts/envs/lm-eval/bin/pip install -r requirements/lm-eval.lock \
    --extra-index-url https://download.pytorch.org/whl/cu121
git clone --branch v0.4.9 --depth 1 \
    https://github.com/EleutherAI/lm-evaluation-harness.git ../artifacts/lm-harness
```

Models, datasets, checkpoints and results are stored in the *artifact root*:
`$SFT_CONFLICT_ARTIFACT_ROOT`, or `../artifacts` next to the repository by default
(`--artifact-root` overrides both). Missing models and datasets are downloaded on first use.

## Quick start

AEGIS on RTE with Mistral-7B:

```bash
# 1. Build the pervasive dataset by label-coverage filtering of SMART FLAN (CPU only).
python run.py --method filtering_anchor --model mistral_7b

# 2. Record the Base model's geometry on the pervasive dataset (one GPU, once per model).
python run.py --method prepare_anchor_geometry_reference --model mistral_7b --gpus 0

# 3. Train with AEGIS.
python run.py --method target_plus_anchor_ce_geometry --model mistral_7b --target rte --gpus 0 1 2 3

# 4. Evaluate on RTE and on the pervasive-capability suite.
python run.py --action native-evaluate --model mistral_7b --target rte \
    --checkpoint ../artifacts/checkpoints/<run-name>/final --llm-evaluation
```

Training prints its run name (`run_name=click-mistral_7b-rte-...`). The selected model is
exported to `<artifact-root>/checkpoints/<run-name>/final` and logs are written to
`<artifact-root>/results/<run-name>/`. Without `--checkpoint`, step 4 evaluates the Base model.

## Training and evaluation for each dataset

Steps 1 and 2 are shared by all tasks of a model. For each task, train with its training target
and evaluate with its evaluation target:

```bash
python run.py --method target_plus_anchor_ce_geometry --model mistral_7b --target <train target>
python run.py --action native-evaluate --model mistral_7b --target <evaluation target> \
    --checkpoint ../artifacts/checkpoints/<run-name>/final
```

| Task | Train target | Evaluation target | Metric |
|---|---|---|---|
| CoLA | `cola` | `cola` | Matthews correlation |
| SST-2 | `sst2` | `sst2` | accuracy |
| MRPC | `mrpc` | `mrpc` | mean of accuracy and F1 |
| STS-B | `stsb` | `stsb` | mean of Pearson and Spearman |
| QQP | `qqp` | `qqp` | mean of accuracy and F1 |
| MNLI | `mnli` | `mnli` | accuracy (matched) |
| QNLI | `qnli` | `qnli` | accuracy |
| RTE | `rte` | `rte` | accuracy |
| GSM8K | `gsm8k` | `gsm8k` | exact match |
| CodeAlpaca | `codealpaca` | `humaneval` | pass@1/5/10 |
| Code Bagel Hermes | `code_bagel_hermes` | `codexglue_python`, `codexglue_java` | token accuracy |
| Infinity-Instruct code | `infinity_instruct_code` | `codexglue_python`, `codexglue_java` | token accuracy |
| Evol-Instruct | `evol_instruct` | `alpaca_eval` | win rate |
| LogiQA 2.0 | `logiqa2` | `logiqa2` | accuracy |

- Add `--llm-evaluation` to any evaluation to also run the pervasive-capability suite.
- HumanEval executes the generated code: add `--allow-code-execution`, preferably on a
  disposable machine.
- AlpacaEval writes the model outputs; `--alpaca-eval-run-judge` also runs the official judge and
  needs `OPENAI_API_KEY`.
- Other models: change `--model` and run steps 1 and 2 for that model first.
- Baselines use the same commands with `--method target_only` (plain SFT) or
  `--method target_plus_anchor_ce` (pervasive data replayed with cross-entropy).

## Create your own pervasive dataset

`create_anchor_dataset` builds a pervasive dataset directly from the five FLAN 2022 submixtures
used by SMART (1,841 tasks):

```bash
python run.py --method create_anchor_dataset --model mistral_7b
```

It visits the tasks in shuffled round-robin order and keeps the samples whose labels add tokens
that are not covered yet, until every sample has been visited. It then prints the number of
samples and tasks, the covered vocabulary and the Base model's accuracy. To use the result, pass
`--anchor flan2022_mistral_7b_task_vocab_cover_v1` in steps 2 and 3. Useful options are
`--creation-seed`, `--creation-submixtures`, `--no-creation-guarantee-task-coverage` and
`--creation-max-samples-per-task`.

## Command-line options

`python run.py --help` lists all options with their defaults. Defaults are defined at the top of
[`run.py`](run.py), so `python run.py` without options runs with them unchanged.

| Option | Meaning |
|---|---|
| `--method` | AEGIS, a baseline, a pipeline step or `test_crossing` (per-token crossing analysis) |
| `--action` | `train` (default) or `native-evaluate` |
| `--model`, `--target`, `--anchor` | model, target task and pervasive dataset (`auto` uses the one from step 1) |
| `--checkpoint`, `--resume` | model to evaluate (`.../final`); `.../checkpoint-N` to resume training |
| `--gpus` | physical GPU ids |
| `--epochs`, `--lr`, `--max-length`, `--seed`, `--max-steps` | training settings |
| `--geometry-config` | AEGIS configuration (`configs/method/anchor_ce_geometry_relative.yaml`) |
| `--set NAME=VALUE` | override any other default defined in `run.py` |
| `--dry-run` | print the resolved command without running it |

## Citation

```bibtex
@misc{chen2026unifyingtheorypervasivecapability,
      title={Unifying and Beyond the Theory of Pervasive Capability Degradation in LLM Post-training SFT from Feature Interference},
      author={Hang Chen and Jiaying Zhu and Wenya Wang},
      year={2026},
      eprint={2611.05003},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2611.05003},
}
```

## Acknowledgements

The pervasive data build on the SMART mixture of FLAN 2022; evaluation uses lm-evaluation-harness
and AlpacaEval. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
