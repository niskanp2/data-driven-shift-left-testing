# Data-Driven Shift-Left Testing via *Should Have Been Found* Statistics Thesis Repository

Repository for master's thesis reproducibility.

## File Structure

```
.
├── data  # Files containing the raw data
├── gen   # Files generated with the jupyter notebooks
├── src   # Files containing source code
├── 01_preprocessing.ipynb       # Generates machine learning data
├── 02_generate_embedding.ipynb  # Generates node embeddings
├── 03_role2vec_em.ipynb         # Trains either sRNN or GRU ML model
└── 04_simulation.ipynb          # Calculates expected cost improvements
```

## Setup

1. Edit uv indices in the `pyproject.toml` file such that the cuda version for pytorch is compatible for your system (you're on your own).
2. Run `uv sync` in the terminal (and install uv if you haven't already) to set up the virtual environment.
3. In the jupyter notebooks, select the `thesis` kernel before running them.

## Running the Code

To reproduce the results in the thesis, run the following files in sequence:

- `01_preprocessing.ipynb`
- `02_generate_embedding.ipynb`
- `03_rol2vec_em.ipynb`
- `04_simulation.ipynb`

> [!NOTE]
> The notebooks can be run out of sequence. The generated files used in the thesis are in the .gen/ directory. Running earlier notebooks will overwrite these resources.
