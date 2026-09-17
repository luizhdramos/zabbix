# %%
# Setup
# Continuação da Data Preparation — df_prime e df_varejo já carregados e com dt_movimento em datetime.
# pip install pandas holidays python-dateutil numpy

import holidays
import numpy as np
import pandas as pd
from dateutil.easter import easter

# %%
# 1. Feriados nacionais OFICIAIS
# A holidays.Brazil() só traz o que é feriado por lei (Ano Novo, Sexta-feira Santa,
# Tiradentes, 1º de Maio, 7 de Setembro, 12/10, 2/11, 15/11, 20/11 a partir de 2024, Natal).

br_holidays = holidays.Brazil(years=range(2019, 2027))

feriados_oficiais = pd.DataFrame(
    [{"ds": pd.Timestamp(data), "holiday": nome} for data, nome in br_holidays.items()]
)

print(f"{len(feriados_oficiais)} feriados oficiais entre 2019 e 2026")

# %%
# 2. Datas móveis do Carnaval (não são feriado oficial, mas o banco para)
# Todas se calculam a partir da Páscoa:
#   Carnaval (segunda) = Páscoa - 48 dias
#   Carnaval (terça)   = Páscoa - 47 dias
#   Quarta de Cinzas   = Páscoa - 46 dias  -> é aqui que o represado é implantado
#   Corpus Christi     = Páscoa + 60 dias

datas_moveis = []

for ano in range(2019, 2027):
    pascoa = pd.Timestamp(easter(ano))
    datas_moveis.append({"ds": pascoa - pd.Timedelta(days=48), "holiday": "carnaval_segunda"})
    datas_moveis.append({"ds": pascoa - pd.Timedelta(days=47), "holiday": "carnaval_terca"})
    datas_moveis.append({"ds": pascoa - pd.Timedelta(days=46), "holiday": "quarta_de_cinzas"})
    datas_moveis.append({"ds": pascoa + pd.Timedelta(days=60), "holiday": "corpus_christi"})

datas_moveis = pd.DataFrame(datas_moveis)

# Conferindo contra o que apareceu nos resíduos do Prime (17/02/21, 02/03/22, 22/02/23, 14/02/24)
print(datas_moveis[datas_moveis["holiday"] == "quarta_de_cinzas"])

# %%
# 3. Recesso de fim de ano
# 25/12 e 01/01 já vêm nos oficiais. Faltam 24/12 (véspera) e 26 a 31/12.
# Evidência: 24/12 apareceu com resíduo forte negativo nos DOIS segmentos, em 2024 e 2025.

recesso = []

for ano in range(2019, 2027):
    recesso.append({"ds": pd.Timestamp(f"{ano}-12-24"), "holiday": "vespera_natal"})
    for dia in range(26, 32):
        recesso.append({"ds": pd.Timestamp(f"{ano}-12-{dia}"), "holiday": "recesso_fim_ano"})

recesso = pd.DataFrame(recesso)

# %%
# 4. Consolidar o calendário final (entrada do Prophet)

holidays_df = (
    pd.concat([feriados_oficiais, datas_moveis, recesso])
    .drop_duplicates(subset="ds", keep="first")   # se colidir, o oficial prevalece
    .sort_values("ds")
    .reset_index(drop=True)
)

print(f"{len(holidays_df)} datas no calendário final")
print(holidays_df["holiday"].value_counts())

# %%
# 5. Flags binárias para o SARIMA — Prime
# O Prophet aceita o holidays_df com um efeito por nome de feriado. O SARIMAX não tem
# esse encolhimento automático, então lá agrupo em 4 flags: se abrisse uma coluna por
# feriado, na janela de 2024 cada um teria só ~2 observações e o coeficiente viria puro ruído.
# Os grupos separam efeitos de SINAL OPOSTO, que é o que não pode ser misturado:

datas_recesso = set(holidays_df.loc[holidays_df["holiday"].isin(["vespera_natal", "recesso_fim_ano"]), "ds"].dt.date)
datas_carnaval = set(holidays_df.loc[holidays_df["holiday"].isin(["carnaval_segunda", "carnaval_terca"]), "ds"].dt.date)
datas_pos_carnaval = set(holidays_df.loc[holidays_df["holiday"] == "quarta_de_cinzas", "ds"].dt.date)
datas_feriado_comum = set(holidays_df["ds"].dt.date) - datas_recesso - datas_carnaval - datas_pos_carnaval

df_prime["is_feriado"] = df_prime["dt_movimento"].dt.date.isin(datas_feriado_comum).astype(int)
df_prime["is_recesso"] = df_prime["dt_movimento"].dt.date.isin(datas_recesso).astype(int)
df_prime["is_carnaval"] = df_prime["dt_movimento"].dt.date.isin(datas_carnaval).astype(int)
df_prime["is_pos_carnaval"] = df_prime["dt_movimento"].dt.date.isin(datas_pos_carnaval).astype(int)

print(df_prime[["is_feriado", "is_recesso", "is_carnaval", "is_pos_carnaval"]].sum())

# %%
# 5b. Flags binárias para o SARIMA — Varejo

df_varejo["is_feriado"] = df_varejo["dt_movimento"].dt.date.isin(datas_feriado_comum).astype(int)
df_varejo["is_recesso"] = df_varejo["dt_movimento"].dt.date.isin(datas_recesso).astype(int)
df_varejo["is_carnaval"] = df_varejo["dt_movimento"].dt.date.isin(datas_carnaval).astype(int)
df_varejo["is_pos_carnaval"] = df_varejo["dt_movimento"].dt.date.isin(datas_pos_carnaval).astype(int)

print(df_varejo[["is_feriado", "is_recesso", "is_carnaval", "is_pos_carnaval"]].sum())

# %%
# 6. Sazonalidade anual para o SARIMA — termos de Fourier (Prime)
# O Prophet modela sazonalidade anual nativamente. O SARIMA não consegue: o termo sazonal
# dele usaria período 365, o que é inviável de estimar. A saída padrão é injetar a "onda"
# anual como coluna exógena: pares de seno/cosseno com período de 1 ano.
# Uso ordem 3 (6 colunas) — o suficiente para uma curva anual suave, sem inflar demais.

EPOCA = pd.Timestamp("2020-01-01")
ORDEM_FOURIER = 3

t_prime = (df_prime["dt_movimento"] - EPOCA).dt.days.values

for k in range(1, ORDEM_FOURIER + 1):
    df_prime[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * t_prime / 365.25)
    df_prime[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * t_prime / 365.25)

df_prime.head()

# %%
# 6b. Sazonalidade anual para o SARIMA — termos de Fourier (Varejo)

t_varejo = (df_varejo["dt_movimento"] - EPOCA).dt.days.values

for k in range(1, ORDEM_FOURIER + 1):
    df_varejo[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * t_varejo / 365.25)
    df_varejo[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * t_varejo / 365.25)

df_varejo.head()

# %%
# 7. Recortar as janelas de novo (agora com as colunas novas) — Prime

FIM_TREINO = pd.Timestamp("2026-07-31")

df_prime_2020 = df_prime[(df_prime["dt_movimento"] >= "2020-01-01") & (df_prime["dt_movimento"] <= FIM_TREINO)]
df_prime_2023 = df_prime[(df_prime["dt_movimento"] >= "2023-01-01") & (df_prime["dt_movimento"] <= FIM_TREINO)]
df_prime_2024 = df_prime[(df_prime["dt_movimento"] >= "2024-01-01") & (df_prime["dt_movimento"] <= FIM_TREINO)]

print("Prime ->", len(df_prime_2020), len(df_prime_2023), len(df_prime_2024))

# %%
# 7b. Recortar as janelas de novo — Varejo

df_varejo_2020 = df_varejo[(df_varejo["dt_movimento"] >= "2020-01-01") & (df_varejo["dt_movimento"] <= FIM_TREINO)]
df_varejo_2023 = df_varejo[(df_varejo["dt_movimento"] >= "2023-01-01") & (df_varejo["dt_movimento"] <= FIM_TREINO)]
df_varejo_2024 = df_varejo[(df_varejo["dt_movimento"] >= "2024-01-01") & (df_varejo["dt_movimento"] <= FIM_TREINO)]

print("Varejo ->", len(df_varejo_2020), len(df_varejo_2023), len(df_varejo_2024))

# %%
# 8. Colunas exógenas do SARIMA (a lista que vai no argumento `exog`)
# O `d` de cada janela já veio do ADF:
#   Prime  -> 2020: d=1 | 2023: d=1 | 2024: d=0
#   Varejo -> 2020: d=1 | 2023: d=1 | 2024: d=1

COLS_EXOG = [
    "is_feriado", "is_recesso", "is_carnaval", "is_pos_carnaval",
    "fourier_sin_1", "fourier_cos_1",
    "fourier_sin_2", "fourier_cos_2",
    "fourier_sin_3", "fourier_cos_3",
]

D_POR_JANELA = {
    ("prime", "2020"): 1, ("prime", "2023"): 1, ("prime", "2024"): 0,
    ("varejo", "2020"): 1, ("varejo", "2023"): 1, ("varejo", "2024"): 1,
}

print(COLS_EXOG)
print(D_POR_JANELA)

# %%
# 9. Conferência: quanto vale, na média, cada grupo de data? (Prime)
# Só para validar se os grupos fazem sentido antes de entrarem no modelo.
# Espero: feriado e recesso ABAIXO da média; pós-carnaval ACIMA.

print("média geral:", round(df_prime_2020["qt_venda"].mean(), 1))
print("feriado comum:", round(df_prime_2020.loc[df_prime_2020["is_feriado"] == 1, "qt_venda"].mean(), 1))
print("recesso:", round(df_prime_2020.loc[df_prime_2020["is_recesso"] == 1, "qt_venda"].mean(), 1))
print("carnaval:", round(df_prime_2020.loc[df_prime_2020["is_carnaval"] == 1, "qt_venda"].mean(), 1))
print("pós-carnaval:", round(df_prime_2020.loc[df_prime_2020["is_pos_carnaval"] == 1, "qt_venda"].mean(), 1))

# %%
# 9b. Conferência — Varejo

print("média geral:", round(df_varejo_2020["qt_venda"].mean(), 1))
print("feriado comum:", round(df_varejo_2020.loc[df_varejo_2020["is_feriado"] == 1, "qt_venda"].mean(), 1))
print("recesso:", round(df_varejo_2020.loc[df_varejo_2020["is_recesso"] == 1, "qt_venda"].mean(), 1))
print("carnaval:", round(df_varejo_2020.loc[df_varejo_2020["is_carnaval"] == 1, "qt_venda"].mean(), 1))
print("pós-carnaval:", round(df_varejo_2020.loc[df_varejo_2020["is_pos_carnaval"] == 1, "qt_venda"].mean(), 1))
