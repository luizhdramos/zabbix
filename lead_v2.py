{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d41995f1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot de conferência — Prime (fica logo depois de resultado_prime_setembro, dentro da 9.0)\n",
    "# Histórico recente encostado na previsão de setembro, pra ver se o nível bate.\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "plt.figure(figsize=(12, 5))\n",
    "plt.plot(df_prime[\"dt_movimento\"], df_prime[\"qt_venda\"], label=\"Real (histórico)\", color=\"steelblue\", linewidth=1)\n",
    "plt.plot(resultado_prime_setembro[\"dt_movimento\"], resultado_prime_setembro[\"previsto\"],\n",
    "         label=\"Previsto (setembro)\", color=\"darkorange\", linewidth=2)\n",
    "plt.fill_between(resultado_prime_setembro[\"dt_movimento\"], resultado_prime_setembro[\"limite_inferior_95\"],\n",
    "                  resultado_prime_setembro[\"limite_superior_95\"], color=\"darkorange\", alpha=0.2, label=\"IC 95%\")\n",
    "plt.axvline(pd.Timestamp(\"2026-06-30\"), color=\"gray\", linestyle=\"--\", linewidth=1, label=\"Fim do treino (30/06)\")\n",
    "plt.xlim(pd.Timestamp(\"2026-01-01\"), pd.Timestamp(\"2026-09-30\"))\n",
    "plt.title(\"Prime — histórico recente x baseline de setembro/2026\")\n",
    "plt.xlabel(\"Data\")\n",
    "plt.ylabel(\"Quantidade de vendas\")\n",
    "plt.legend()\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c8bc54cb",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot de conferência — Varejo (fica logo depois de resultado_varejo_setembro, dentro da 9.0)\n",
    "\n",
    "plt.figure(figsize=(12, 5))\n",
    "plt.plot(df_varejo[\"dt_movimento\"], df_varejo[\"qt_venda\"], label=\"Real (histórico)\", color=\"steelblue\", linewidth=1)\n",
    "plt.plot(resultado_varejo_setembro[\"dt_movimento\"], resultado_varejo_setembro[\"previsto\"],\n",
    "         label=\"Previsto (setembro)\", color=\"seagreen\", linewidth=2)\n",
    "plt.fill_between(resultado_varejo_setembro[\"dt_movimento\"], resultado_varejo_setembro[\"limite_inferior_95\"],\n",
    "                  resultado_varejo_setembro[\"limite_superior_95\"], color=\"seagreen\", alpha=0.2, label=\"IC 95%\")\n",
    "plt.axvline(pd.Timestamp(\"2026-06-30\"), color=\"gray\", linestyle=\"--\", linewidth=1, label=\"Fim do treino (30/06)\")\n",
    "plt.xlim(pd.Timestamp(\"2026-01-01\"), pd.Timestamp(\"2026-09-30\"))\n",
    "plt.title(\"Varejo — histórico recente x baseline de setembro/2026\")\n",
    "plt.xlabel(\"Data\")\n",
    "plt.ylabel(\"Quantidade de vendas\")\n",
    "plt.legend()\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "54a8bd64",
   "metadata": {},
   "source": [
    "## 10.0 Visualização\n",
    "\n",
    "As duas plotagens que ficam dentro da 9.0 (logo depois de cada tabela de segmento) são de **conferência**: mostram o histórico recente encostado na previsão de setembro, com uma linha vertical marcando onde o treino parou (30/06) — servem pra você ver de olho se o nível da previsão faz sentido continuando a série real, sem ficar só confiando na tabela de números.\n",
    "\n",
    "Essa seção fecha com a versão **limpa**, só de setembro, Prime e Varejo lado a lado — essa é a que interessa pro time de negócio, porque é exatamente o recorte que vai ser comparado com o dado real quando setembro fechar (sem os 6 anos de histórico atrás, que só atrapalhariam a leitura nesse momento).\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6563f210",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 10.0 Visualização — versão limpa, só setembro, lado a lado (Prime x Varejo)\n",
    "# As duas plotagens acima servem de conferência (histórico + previsão, pra ver se o nível bate).\n",
    "# Essa aqui é a versão pro time de negócio: só o mês de setembro, sem os 6 anos de histórico atrás,\n",
    "# porque é exatamente o que vai ser comparado com o dado real quando setembro fechar.\n",
    "\n",
    "fig, eixos = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "eixos[0].plot(resultado_prime_setembro[\"dt_movimento\"], resultado_prime_setembro[\"previsto\"],\n",
    "              color=\"darkorange\", linewidth=2, marker=\"o\", markersize=3, label=\"Previsto\")\n",
    "eixos[0].fill_between(resultado_prime_setembro[\"dt_movimento\"], resultado_prime_setembro[\"limite_inferior_95\"],\n",
    "                       resultado_prime_setembro[\"limite_superior_95\"], color=\"darkorange\", alpha=0.2, label=\"IC 95%\")\n",
    "eixos[0].set_title(\"Prime — baseline de setembro/2026\")\n",
    "eixos[0].set_xlabel(\"Data\")\n",
    "eixos[0].set_ylabel(\"Quantidade de vendas\")\n",
    "eixos[0].tick_params(axis=\"x\", rotation=45)\n",
    "eixos[0].legend()\n",
    "\n",
    "eixos[1].plot(resultado_varejo_setembro[\"dt_movimento\"], resultado_varejo_setembro[\"previsto\"],\n",
    "              color=\"seagreen\", linewidth=2, marker=\"o\", markersize=3, label=\"Previsto\")\n",
    "eixos[1].fill_between(resultado_varejo_setembro[\"dt_movimento\"], resultado_varejo_setembro[\"limite_inferior_95\"],\n",
    "                       resultado_varejo_setembro[\"limite_superior_95\"], color=\"seagreen\", alpha=0.2, label=\"IC 95%\")\n",
    "eixos[1].set_title(\"Varejo — baseline de setembro/2026\")\n",
    "eixos[1].set_xlabel(\"Data\")\n",
    "eixos[1].tick_params(axis=\"x\", rotation=45)\n",
    "eixos[1].legend()\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
