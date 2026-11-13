# Model comparison

Selected for registry alias `champion`: **champion** (catboost)

Rule: best validation f1 among models with validation recall >= 0.7.

| Model | Framework | Split | F1 | Recall | Precision | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| baseline | lightgbm | val | 0.5976 | 0.7082 | 0.5169 | 0.8152 | 0.6022 |
| baseline | lightgbm | test | 0.6023 | 0.7464 | 0.5048 | 0.8344 | 0.6430 |
| champion | catboost | val | 0.6168 | 0.7046 | 0.5485 | 0.8287 | 0.6390 |
| champion | catboost | test | 0.6424 | 0.7571 | 0.5579 | 0.8467 | 0.6589 |
