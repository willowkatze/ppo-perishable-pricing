# Raw Data

Raw datasets are not committed to this public repository.

## Primary Dataset

FreshRetailNet-50K by Dingdong-Inc is the primary operational dataset used by the final project. Download it separately from Hugging Face according to the dataset license and terms:

```text
Dingdong-Inc/FreshRetailNet-50K
```

Expected local raw files may include:

```text
data/operational/raw/freshretail/raw/freshretail_train.parquet
data/operational/raw/freshretail/raw/freshretail_eval.parquet
```

## Derived Artifacts

Processed data and fitted model artifacts are also excluded from the public Git archive by default. See:

- `docs/data_and_assumptions.md`
- `docs/reproducibility.md`
- `outputs/models/MODEL_ARTIFACTS.md`

The repository keeps source code, locked configs, selected result tables, report-ready figures, and checksum manifests, but not raw or restricted datasets.
