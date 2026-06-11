# Latent Lingua Splines

Watch a language model think. As an LLM generates each token, its internal state travels through a high-dimensional embedding space. This project captures that journey, projects it to 3-D via **PCA → UMAP**, and renders it as an interactive animated spline you can explore token by token.

```
question → LLM generates token-by-token
            ↓
            for each token: embedding vector  (+ hidden states per layer, optional)
            ↓
            PCA 512-D  →  UMAP 3-D
            ↓
            animated spline  ·  interactive controls  ·  hover linking
```

---

## Installation

```bash
pip install -r requirements.txt
```

> **Python 3.10+** required. For GPU inference install a CUDA-enabled PyTorch build first, then run `pip install -r requirements.txt`.

---

## Quickstart — three steps

### Step 1 — Fit the projection (once)

Fit PCA + UMAP on the model's vocabulary embeddings. Saves to `models/projection.pkl` and is reused by all other scripts.

```bash
python scripts/train_projection.py
```

```bash
python scripts/train_projection.py --no-viz        # skip the static plot
python scripts/train_projection.py --config config/config.yaml
```

> This step takes a few minutes the first time. After that every other script loads the saved projection instantly.

---

### Step 2 — Explore the vocabulary manifold (optional)

Interactive 3-D viewer of the entire token vocabulary. Coordinates are cached to `outputs/coords_cache.npz` so subsequent launches are instant.

```bash
python scripts/visualize_space.py
python scripts/visualize_space.py --recompute      # force re-extraction if the model changed
python scripts/visualize_space.py --config config/config.yaml
```

**Controls**

| Action | How |
|--------|-----|
| Rotate | Left-drag |
| Zoom | Scroll wheel |
| Inspect token | Hover — nearest token label appears |
| Highlight tokens | Type a regex in the search box → Enter |
| Clear highlight | Click **Clear highlight** |
| Change colour mode | Radio buttons: Azimuth / Elevation / Length / Type |

---

### Step 3 — Animate a query

Watch the model generate a response as a trajectory through the latent space.

```bash
# Basic usage (manual step mode by default)
python scripts/query_trajectory.py "What is gravity?"

# Auto-playing animation
python scripts/query_trajectory.py --auto "What is gravity?"

# Interactive prompt (no argument)
python scripts/query_trajectory.py
```

**All flags**

| Flag | Default | Description |
|------|---------|-------------|
| `--auto` | off | Auto-play the animation. Without this flag the animation is paused and you step through it manually. |
| `--add-hidden-states` | off | Project hidden states from every transformer layer (richer trajectory). Default uses only token embedding vectors — faster and lighter. |
| `--follow` | off | Camera tracks the arrowhead through the space. |
| `--ms-per-token` | `300` | Animation time per token in ms (auto mode only). |
| `--max-tokens` | `80` | Maximum tokens to generate. |
| `--config` | `config/config.yaml` | Path to config file. |

---

## Interactive controls — trajectory viewer

### Manual step mode (default)

When `--auto` is **not** passed the animation starts paused at the first token. Use the controls below to walk through the response.

| Action | Control |
|--------|---------|
| Next token | → arrow key  or  **D**  or  **Next >>** button |
| Previous token | ← arrow key  or  **A**  or  **<< Prev** button |

A thin **progress bar** above the buttons shows how far through the response you are. The **token counter** between the buttons shows the current position (`K / N`). The response text on the right reveals tokens as you advance.

### Auto mode (`--auto`)

The animation plays continuously. Use the mouse to rotate, zoom, and hover while it plays.

| Action | How |
|--------|-----|
| Rotate | Left-drag |
| Zoom | Scroll wheel |
| Inspect token | Hover over any coloured word in the response panel — the matching 3-D dot glows and enlarges |

---

## Two trajectory modes

### Embedding-only (default)

Uses only the token embedding lookup vector for each generated token. No hidden states are captured during generation.

- **Fast** — no extra memory or compute during generation.
- Trajectory has **N control points** (one per token).
- Good for understanding *what* tokens the model chose and *where* they sit in the vocabulary manifold.

```bash
python scripts/query_trajectory.py "Describe the sea"
```

### With hidden states (`--add-hidden-states`)

Captures the residual-stream vector after every transformer block for each token. The trajectory passes through all `N × L` positions.

- **Slower** — requires `output_hidden_states=True` during generation.
- Trajectory has **N × L control points** — one per (token, layer) pair.
- Intermediate layer dots (smaller, 70% opacity) appear between each token's anchor dot, revealing how the model's internal representation evolves through depth.
- Good for understanding *how* the model builds up each token's representation layer by layer.

```bash
python scripts/query_trajectory.py --add-hidden-states "Describe the sea"
```

---

## Configuration

All parameters live in `config/config.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `model.name` | `Qwen/Qwen3.5-0.8B` | Any HuggingFace causal LM |
| `projection_samples` | `-1` | Vocab tokens used to fit PCA+UMAP (−1 = all) |
| `pca.n_components` | `512` | Intermediate PCA dimensionality |
| `umap.n_components` | `3` | Output dimensionality (keep at 3) |
| `umap.n_neighbors` | `15` | UMAP neighbourhood size |
| `umap.min_dist` | `0.1` | UMAP minimum distance |
| `umap.metric` | `cosine` | Distance metric |
| `umap.n_jobs` | `12` | Parallel threads for UMAP fitting (−1 = all cores) |
| `umap.random_state` | `42` | RNG seed for reproducibility — ignored when `n_jobs ≠ 1` (UMAP requires single-threaded mode for a fixed seed) |
| `projection.output_path` | `models/projection.pkl` | Where to save the fitted projection |
| `visualization.output_path` | `outputs/embedding_space.png` | Static plot output path |

---

## Project structure

```
├── config/
│   └── config.yaml                  # all parameters
├── src/
│   ├── embeddings.py                # EmbeddingExtractor — vocab embeddings from any HF model
│   ├── projection.py                # LatentProjection  — PCA → UMAP fit / transform / save
│   └── utils.py                     # load_config helpers
├── scripts/
│   ├── train_projection.py          # fit and save the PCA+UMAP projection
│   ├── visualize_space.py           # interactive vocabulary manifold viewer
│   └── query_trajectory.py          # animated LLM response trajectory
├── data/
│   └── good_prompts.txt             # sample prompts to try
├── models/
│   └── projection.pkl               # saved projection (created by train_projection.py)
└── outputs/
    ├── embedding_space.png          # static manifold plot
    └── coords_cache.npz             # cached 3-D token coordinates (created by visualize_space.py)
```

---

## Supported models

Any HuggingFace causal LM works. The embedding layer is located automatically:

| Architecture | Examples |
|---|---|
| Qwen3 / Qwen2 | `Qwen/Qwen3-0.6B`, `Qwen/Qwen2.5-1.5B` |
| LLaMA / Mistral / Falcon | `meta-llama/Llama-3.2-1B`, `mistralai/Mistral-7B-v0.1` |
| GPT-2 / GPT-Neo | `gpt2`, `EleutherAI/gpt-neo-125m` |

If a model's architecture is not explicitly recognised, the extractor scans all `nn.Embedding` layers and picks the one with the largest vocabulary. Hidden states are captured via HuggingFace's `output_hidden_states=True`, so any model that supports `model.generate()` will work.

---

## License

MIT — see [LICENSE](LICENSE).
