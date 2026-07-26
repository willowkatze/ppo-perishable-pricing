# Raw Data

Raw datasets are not committed to this public repository.

## Primary Dataset

FreshRetailNet-50K by Dingdong-Inc is the primary operational dataset used by the final project. Download it separately from Hugging Face according to the dataset license and terms:

```text
Dingdong-Inc/FreshRetailNet-50K
```

The public dataset is described at a high level as covering approximately three months (approximately 90 days). The complete sequences selected for this project contain exactly 97 daily observations from 2024-03-28 through 2024-07-02.

Expected local raw files may include:

```text
data/operational/raw/freshretail/raw/freshretail_train.parquet
data/operational/raw/freshretail/raw/freshretail_eval.parquet
```

## Derived Artifacts

Processed data and fitted model artifacts are also excluded from the public Git archive by default. See:

- [Data and demand recovery](../../docs/02_data_and_demand_recovery.md)
- [Environment and methods](../../docs/03_environment_and_methods.md)
- [Reproducibility](../../docs/reproducibility.md)
- [Local model artifacts](../../outputs/models/MODEL_ARTIFACTS.md)

The repository keeps source code, locked configs, selected result tables, report-ready figures, and checksum manifests, but not raw or restricted datasets.
