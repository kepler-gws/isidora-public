[![DOI](https://zenodo.org/badge/1409519846.svg)](https://doi.org/10.5281/zenodo.23247249)

>...He was thinking of all these things when he desired a city. Isidora, therefore, is the city of his dreams: with one difference. The dreamed-of city contained him as a young man; he arrives at Isidora in his old age. In the square there is the wall where the old men sit and watch the young go by; he is seated in a row with them. Desires are already memories.
>
> -Italo Calvino, *Invisible Cities*

Publicly available code to generate data and figures for [arXiv:2609.14130](https://arxiv.org/abs/2609.14130).

# Reproducing paper figures

In order:

1. `generate-training-set.py`
2. `full_tile_rb.sh`
3. Using `maestrowf`, run `pipeline-test-splits.yaml` with the associated `-pgen.py` file.
4. `Generate Toy MCMC Problem.ipynb`
5. `Run Toy MCMC Problem.ipynb`
6. `Paper Figures.ipynb`
7. Optionally, verify that using RB + EIM waveform is equivalent to ROQ likelihood in `ROQ Likelihood Verification.ipynb`.
