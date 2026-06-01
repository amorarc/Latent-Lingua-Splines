# Latent Lingua Splines

Visualise how phrases travel through a language model's latent space.

Each token in a phrase maps to a point in the model's static embedding table. By projecting those points through **PCA → UMAP**, we reduce them to 2-D (or 3-D) coordinates and draw the ordered sequence as a spline — the *trajectory* the phrase traces through the learned latent geometry.

```
"the quick brown fox"
  ↓ token embeddings (768-D)
  ↓ PCA  →  50-D
  ↓ UMAP →   2-D
  → spline connecting [the] → [quick] → [brown] → [fox]
```

## Why PCA + UMAP?

- **PCA first** collapses the high-dimensional redundancy linearly, reducing noise and making UMAP's job easier.
- **UMAP second** captures the nonlinear manifold structure that PCA misses.

The two-stage pipeline consistently produces cleaner, more interpretable trajectories than either method alone.

## Installation

```bash
pip install -r requirements.txt
```

> Requires Python 3.10+. For GPU acceleration, install a CUDA-enabled PyTorch build before running pip install.

## Quick start

1. **Add your phrases** — create `data/corpus.txt` with one phrase per line:

   ```
   the quick brown fox jumps over the lazy dog
   language models learn rich representations
   attention is all you need
   ```

2. **Configure the model** — edit `config/config.yaml`:

   ```yaml
   model:
     name: "bert-base-uncased"   # any HuggingFace model name
   ```

3. **Train the projection**:

   ```bash
   python scripts/train_projection.py
   ```

   This fits PCA + UMAP on the corpus, saves the projection to `models/projection.pkl`, and writes a trajectory plot to `outputs/trajectories.png`.

4. **Skip the plot** if you only need the saved model:

   ```bash
   python scripts/train_projection.py --no-viz
   ```

## Configuration reference

All parameters live in `config/config.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `model.name` | `bert-base-uncased` | Any HuggingFace model identifier |
| `data.corpus_path` | `data/corpus.txt` | Plain-text corpus, one phrase per line |
| `data.max_phrases` | `null` | Subsample the corpus (null = use all) |
| `pca.n_components` | `50` | Intermediate PCA dimensionality |
| `umap.n_components` | `3` | Final dimensionality (2 or 3) |
| `umap.n_neighbors` | `15` | UMAP neighbourhood size |
| `umap.min_dist` | `0.1` | UMAP minimum distance |
| `umap.metric` | `cosine` | Distance metric |
| `projection.output_path` | `models/projection.pkl` | Where to save the fitted projection |
| `visualization.output_path` | `outputs/trajectories.png` | Trajectory plot output path |
| `visualization.max_trajectories` | `20` | Phrases to draw in the plot |

## Supported models

The embedding extractor automatically detects the token embedding layer for most HuggingFace architectures:

| Architecture | Examples |
|---|---|
| BERT / RoBERTa / DistilBERT | `bert-base-uncased`, `roberta-base` |
| GPT-2 / GPT-Neo / GPT-J | `gpt2`, `EleutherAI/gpt-neo-125m` |
| T5 | `t5-small`, `google/flan-t5-base` |
| LLaMA / Mistral / Falcon | `meta-llama/Llama-2-7b-hf`, `mistralai/Mistral-7B-v0.1` |

If a model is not recognized, an error message will tell you exactly which attribute path to add in [src/embeddings.py](src/embeddings.py).

## Project structure

```
├── config/
│   └── config.yaml          # all parameters
├── src/
│   ├── embeddings.py        # EmbeddingExtractor — loads LM, extracts token vectors
│   ├── projection.py        # LatentProjection  — PCA → UMAP fit / transform / save
│   └── utils.py             # load_config, load_corpus
├── scripts/
│   └── train_projection.py  # end-to-end training + visualisation script
├── data/                    # place corpus.txt here
├── models/                  # trained projection saved here
└── outputs/                 # trajectory plots saved here
```

## License

MIT — see [LICENSE](LICENSE).
