# Data and Assumptions

The project uses FreshRetailNet-50K as the source of operational sales, promotion, and stockout information. Raw data are excluded from the repository and must be downloaded separately from Hugging Face.

Recovered demand is a model-based estimate, not ground truth. Perishability, waste, batch aging, and financial costs are semi-synthetic because the source data do not contain complete batch-level expiration, realized waste, replenishment, disposal, and cost accounting fields.

Public GitHub archive rule: keep source code, documentation, selected tables, report-ready figures, configs, manifests, and hashes; exclude raw datasets and large generated data artifacts.
