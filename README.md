# selbound

**Response-adjusted selection bounds for single-cell perturbation data.**

Comparing treated with untreated single-cell transcriptomes is the standard way to ask whether
drug-tolerant cells were *selected* (pre-existing states that survived) or *induced* (cells
reprogrammed by the drug). The comparison cannot separate the two on its own: a drug changes
cell state even when survival is completely nonselective.

`selbound` answers a narrower, falsifiable question:

> After allowing a response shared by all cells, how much pre-treatment enrichment does the
> treated population still require, and is that more than survival can produce?

It returns a finite-sample lower confidence bound `M_L` on the selective enrichment
`M*(ν | μK) = exp D∞(ν ‖ μK)` that remains after a held-out response model `K`, and compares it
with a selection budget derived from survival and division. If `M_L` exceeds the budget,
*selection plus a shared response* cannot explain the data.

## Install

```bash
pip install selbound            # numpy, scipy, pandas, scikit-learn
pip install "selbound[anndata]"   # optional AnnData helper
```

## Quick start

```python
import selbound as sb

# Y: cells x dims embedding (e.g. a PCA fitted on held-out FIT cells), groups: sample labels
res = sb.assess_selection(Y, groups, reference="untreated", target="day3",
                          models=("identity", "translation"),
                          alpha=0.05, n_tests=20,          # familywise over everything you report
                          budget=sb.survival_budget(max_divisions=3, frac_surviving=0.46))
print(res.summary())
# day3 vs untreated: M_L (identity 40.95, translation 5.60); statistic 5.60 vs budget 17.30
#   -> compatible with selection + shared response
```

With AnnData:

```python
res = sb.assess_anndata(adata, groupby="sample", reference="0", target="3_rep1",
                        use_rep="X_pca", n_dims=20, budget=17.3)
```

`res.bounds` holds the bound for each response model, `res.statistic` the minimum over the
non-identity models (a valid test statistic without multiplicity correction across models),
`res.loss` the reverse-direction bound (loss of baseline states, which selection can produce)
and `res.ceiling` the largest bound the sample sizes allow.

## What you need

| Input | Why | How to get it |
|---|---|---|
| Baseline and treated cells in one embedding | The bound compares two distributions | Thin to equal depth (`sb.thin`), fit the embedding on FIT cells only (`sb.PearsonResidualPCA`) |
| A FIT/EVAL split | The response model must be fitted on cells the bound does not use | `sb.split_fit_eval`, or your own mask |
| A response class, fixed in advance | Defines what "shared response" means | `translation` (default, calibrated), `axis_stretch`, `affine` (sensitivity only), or any fitted map via `sb.Callable` |
| A selection budget | Defines what selection can produce | `sb.survival_budget(d, f)` from division counts and surviving fraction; `sb.proxy_budget` from tracked lineages |
| Lineage barcodes (optional) | Decomposes the mean shift into selection between clones and within-clone response | `sb.clone_decomposition`, with its no-plasticity floor from `sb.calibrate_floor` |

## Choosing the response class

The class decides what the test can reject. A model flexible enough to map baseline onto treated
exactly (any Markov kernel, or a full affine moment-matching map) always returns `M_L ≈ 1`: the
test stays valid but loses its power. Choose the class before looking at the bound, by external
predictive criteria, and calibrate it on your data (see `examples/`). In the paper, one common
translation was valid (0/60 false rejections) and detected large clone-specific and cell-level
departures (60/60).

## Interpreting the result

* `statistic > budget`: selection plus a response in the class is rejected at level `alpha`.
* `statistic ≤ budget`: the data are *compatible* with selection plus a shared response. This
  does not show that selection operates.
* A rejection survives technical batch inflation smaller than `sb.batch_tolerance(statistic, budget)`.
  Compare two replicate libraries with `sb.lower_bound` to see how large that inflation is in
  your data.
* `selbound` complements differential-abundance tools such as Milo: Milo asks *where*
  abundance changes; `selbound` asks whether the unexplained magnitude exceeds what selection
  can produce.

## Reproducing the paper

`examples/reproduce_pc9.py` rebuilds Table 1 of the paper from the public GEO files of
GSE150949 (identical to the published numbers).

## Citation

Alvarez I. Response-adjusted selection bounds distinguish pre-treatment selection from induced
state change in single-cell perturbation data (2026). See `CITATION.cff`.

## License

MIT
