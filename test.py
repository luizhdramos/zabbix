import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    PrecisionRecallDisplay,
    roc_auc_score,
    roc_curve,
    RocCurveDisplay
)

# -------------------------------------------------------------------------
# 1. Exemplo de dados (Substitua pelas suas variáveis reais)
# y_true: labels reais (0 ou 1)
# y_scores: probabilidades preditas pelo seu modelo (ex: model.predict_proba(X)[:, 1])
# -------------------------------------------------------------------------
# Criando dados fictícios apenas para o código rodar
np.random.seed(42)
y_true = np.random.randint(0, 2, size=1000)
y_scores = np.random.uniform(0, 1, size=1000)

# -------------------------------------------------------------------------
# 2. Cálculo das Métricas
# -------------------------------------------------------------------------
# O average_precision_score calcula o AUC-PR de forma robusta
auc_pr = average_precision_score(y_true, y_scores)
auc_roc = roc_auc_score(y_true, y_scores)

print(f"ROC AUC Score: {auc_roc:.4f}")
print(f"PR AUC Score : {auc_pr:.4f}")

# -------------------------------------------------------------------------
# 3. Plotagem das Curvas (Lado a Lado para Comparação)
# -------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot da Curva ROC
RocCurveDisplay.from_predictions(y_true, y_scores, ax=ax1, color="darkorange")
ax1.plot([0, 1], [0, 1], "k--", label="Classificador Aleatório (AUC = 0.50)")
ax1.set_title("Curva ROC")
ax1.set_xlabel("Taxa de Falsos Positivos (FPR)")
ax1.set_ylabel("Taxa de Verdadeiros Positivos (TPR / Recall)")
ax1.legend(loc="lower right")
ax1.grid(True, linestyle="--", alpha=0.7)

# Plot da Curva Precision-Recall
PrecisionRecallDisplay.from_predictions(y_true, y_scores, ax=ax2, color="blue")
# A linha de base do PR é a proporção de positivos na base
linha_base_pr = np.sum(y_true) / len(y_true)
ax2.plot([0, 1], [linha_base_pr, linha_base_pr], "k--", label=f"Linha de Base (Proporção = {linha_base_pr:.2f})")
ax2.set_title("Curva Precision-Recall (PR)")
ax2.set_xlabel("Revocação (Recall / TPR)")
ax2.set_ylabel("Precisão (Precision)")
ax2.legend(loc="lower left")
ax2.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()
