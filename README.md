# Forge

**Self-Improving Code & Math Reasoning via Verified Self-Play Fine-Tuning**

> One H100 40GB. Twenty-four hours. No reward model. No human annotators.

Forge trains a 7B language model to substantially improve at code generation and mathematical reasoning by learning from its own verified outputs through an iterative self-play loop.

## Key Idea

For domains with **verifiable correctness** — code (does it pass tests?) and math (is the answer right?) — the world itself is the reward model:

1. **Generate** K candidate solutions per problem (temperature sampling)
2. **Verify** each solution (execute code / check math answers)
3. **Build preference pairs** automatically (correct > incorrect)
4. **Train with DPO** on self-generated, self-verified preference data
5. **Iterate**: improved model → better solutions → better training data

## Architecture Highlights

### Stratified LoRA

Depth-aware adapter architecture that allocates more trainable parameters to reasoning-heavy layers:

| Layer Group    | Layers | Rank | Purpose                                |
| -------------- | ------ | ---- | -------------------------------------- |
| Foundation     | 0–6    | 8    | Language understanding (mostly frozen) |
| Reasoning Core | 7–13   | 32   | Algorithmic planning                   |
| Deep Reasoning | 14–20  | 48   | Multi-step computation                 |
| Generation     | 21–27  | 24   | Output synthesis                       |

### 3-Phase Training Pipeline

```
Phase 1: Reasoning SFT     (Hours  0–6)   → Foundation
Phase 2: Self-Play DPO     (Hours  6–20)  → Self-improvement
Phase 3: Refinement        (Hours 20–24)  → Push limits
```

### Multi-Signal Preference Pairs

- **Correctness**: correct solution > incorrect solution
- **Reasoning quality**: coherent steps > jumbled reasoning
- **Conciseness**: shorter correct > longer correct

## Quick Start

### Prerequisites

- NVIDIA H100/A100 40GB (or equivalent) {We have used H100 for this experiment}
- Python 3.10+
- Docker (optional, for code sandbox)

### Installation

```bash
pip install -e ".[eval]"
```

### Run Full Pipeline

```bash
python run_forge.py --config config.yaml
```

### Run Individual Phases

```bash
# Phase 1: SFT
python -m scripts.phase1_sft --config config.yaml

# Phase 2: Self-Play Generation
python -m scripts.phase2_generate --config config.yaml \
    --adapter-path ./checkpoints/phase1_sft/final

# Phase 2: DPO Training
python -m scripts.phase2_dpo --config config.yaml \
    --adapter-path ./checkpoints/phase1_sft/final \
    --solutions-cache ./cache/selfplay_solutions

# Phase 3: Refinement
python -m scripts.phase3_refine --config config.yaml \
    --dpo-adapter-path ./checkpoints/phase2_dpo/final \
    --sft-adapter-path ./checkpoints/phase1_sft/final

# Evaluation
python -m scripts.evaluate --config config.yaml \
    --adapter-path ./checkpoints/phase3_final/final \
    --mode full
```

## Project Structure

```
Limits/
├── config.yaml              # Central configuration
├── run_forge.py              # Main orchestrator
├── pyproject.toml            # Dependencies
├── forge/                    # Core library
│   ├── config.py             # Configuration loader
│   ├── lora.py               # Stratified LoRA implementation
│   ├── data/
│   │   ├── datasets.py       # Dataset loading & formatting
│   │   ├── pair_builder.py   # Preference pair construction
│   │   └── curriculum.py     # Difficulty scheduling
│   └── verify/
│       ├── code_sandbox.py   # Docker-based code execution
│       ├── math_verifier.py  # Symbolic math verification
│       └── process_verifier.py # Step-level reasoning analysis
└── scripts/
    ├── phase1_sft.py         # Supervised fine-tuning
    ├── phase2_generate.py    # Self-play solution generation
    ├── phase2_dpo.py         # DPO training
    ├── phase3_refine.py      # Iterative refinement
    └── evaluate.py           # Benchmark evaluation
```

## Hardware Requirements

| Component | Minimum   | Recommended |
| --------- | --------- | ----------- |
| GPU       | H100 40GB | H100 80GB   |
| RAM       | 32 GB     | 64 GB       |
| Storage   | 100 GB    | 200 GB SSD  |
| Time      | ~24 hours | ~24 hours   |

## Configuration

All settings live in `config.yaml`. Key parameters:

- **Model**: Base model, quantization, sequence length
- **Stratified LoRA**: Per-layer-group rank and alpha values
- **Training**: Learning rates, batch sizes, epochs for each phase
- **Self-Play**: Number of candidates (K), temperature, verification settings
- **Curriculum**: Difficulty thresholds and phase-specific mixes

## Expected Results

| Benchmark | Baseline | Post-Forge (Conservative) |
| --------- | -------- | ------------------------- |
| GSM8K     | ~84%     | ~88%                      |
| MATH      | ~50%     | ~57%                      |
| HumanEval | ~88%     | ~90%                      |
| MBPP+     | ~76%     | ~80%                      |

## License

Apache 2.0
