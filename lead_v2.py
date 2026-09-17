# %%
# Setup (continuação)
# Este bloco continua o notebook da Data Preparation (células 1 a 4).
# pandas e holidays já foram importados lá; aqui entram só os novos.
# pip install numpy python-dateutil

import numpy as np
from dateutil.easter import easter

# %%
# 5. Calendário completo — feriados oficiais
# Reaproveita o br_holidays criado na célula 1. Aqui só separo num DataFrame próprio,
# porque o calendário final vai ganhar mais datas nas próximas células.

feriados_oficiais = pd.DataFrame(
    [{"ds": pd.Timestamp(data), "holiday": nome} for data, nome in br_holidays.items()]
)

print(f"{len(feriados_oficiais)} feriados oficiais entre 2019 e 2026")
feriados_oficiais.head()

# %%
# 5. Calendário completo — datas móveis do Carnaval
# Carnaval e Corpus Christi são ponto facultativo, não feriado por lei — a holidays.Brazil()
# não traz nenhum dos dois. Mas o banco para, então a série sente.
# Todas saem da Páscoa:
#   Carnaval (segunda) = Páscoa - 48  |  Carnaval (terça) = Páscoa - 47
#   Quarta de Cinzas   = Páscoa - 46  |  Corpus Christi   = Páscoa + 60

pascoas = pd.to_datetime([easter(ano) for ano in range(2019, 2027)])

carnaval_segunda = pascoas - pd.Timedelta(days=48)
carnaval_terca = pascoas - pd.Timedelta(days=47)
quarta_de_cinzas = pascoas - pd.Timedelta(days=46)
corpus_christi = pascoas + pd.Timedelta(days=60)

# Confere contra os resíduos do Prime: 17/02/21, 02/03/22, 22/02/23, 14/02/24
print(quarta_de_cinzas)

# %%
# 5. Calendário completo — recesso de fim de ano
# 25/12 e 01/01 já vêm nos oficiais. Faltam a véspera (24/12) e o miolo (26 a 31/12).
# Evidência: 24/12 deu resíduo forte negativo nos dois segmentos, em 2024 e 2025.

todos_os_dias = pd.date_range("2019-01-01", "2026-12-31", freq="D")

vespera_natal = todos_os_dias[(todos_os_dias.month == 12) & (todos_os_dias.day == 24)]
recesso_fim_ano = todos_os_dias[(todos_os_dias.month == 12) & (todos_os_dias.day >= 26)]

print("vésperas:", len(vespera_natal), "| dias de recesso:", len(recesso_fim_ano))

# %%
# 5. Calendário completo — consolidar
# ATENÇÃO: isto substitui o holidays_df que a célula 1 tinha criado.
# Os oficiais entram primeiro, então em caso de data repetida o nome oficial prevalece.

holidays_df = pd.concat([
    feriados_oficiais,
    pd.DataFrame({"ds": carnaval_segunda, "holiday": "carnaval_segunda"}),
    pd.DataFrame({"ds": carnaval_terca, "holiday": "carnaval_terca"}),
    pd.DataFrame({"ds": quarta_de_cinzas, "holiday": "quarta_de_cinzas"}),
    pd.DataFrame({"ds": corpus_christi, "holiday": "corpus_christi"}),
    pd.DataFrame({"ds": vespera_natal, "holiday": "vespera_natal"}),
    pd.DataFrame({"ds": recesso_fim_ano, "holiday": "recesso_fim_ano"}),
]).drop_duplicates(subset="ds", keep="first").sort_values("ds").reset_index(drop=True)

# holidays_df entra direto no Prophet: Prophet(holidays=holidays_df)
print(f"{len(holidays_df)} datas no calendário final")
holidays_df["holiday"].value_counts()

# %%
# 6. Flags de feriado pro SARIMA — Prime
# O Prophet usa o holidays_df direto, com um efeito estimado por nome de feriado.
# O SARIMAX não tem o encolhimento automático do Prophet: se eu abrisse uma coluna por
# feriado, na janela de 2024 cada um teria ~2 observações e o coeficiente viria puro ruído.
# Então agrupo em 4 flags, separando o que tem sinal oposto — feriado e recesso puxam
# pra baixo, pós-carnaval puxa pra cima; juntar os dois anularia o efeito.
# ATENÇÃO: o is_feriado aqui substitui o da célula 1, que juntava tudo numa flag só.

datas_recesso = set(holidays_df.loc[holidays_df["holiday"].isin(["vespera_natal", "recesso_fim_ano"]), "ds"].dt.date)
datas_carnaval = set(holidays_df.loc[holidays_df["holiday"].isin(["carnaval_segunda", "carnaval_terca"]), "ds"].dt.date)
datas_pos_carnaval = set(holidays_df.loc[holidays_df["holiday"] == "quarta_de_cinzas", "ds"].dt.date)
datas_feriado_comum = set(holidays_df["ds"].dt.date) - datas_recesso - datas_carnaval - datas_pos_carnaval

df_prime["is_feriado"] = df_prime["dt_movimento"].dt.date.isin(datas_feriado_comum).astype(int)
df_prime["is_recesso"] = df_prime["dt_movimento"].dt.date.isin(datas_recesso).astype(int)
df_prime["is_carnaval"] = df_prime["dt_movimento"].dt.date.isin(datas_carnaval).astype(int)
df_prime["is_pos_carnaval"] = df_prime["dt_movimento"].dt.date.isin(datas_pos_carnaval).astype(int)

df_prime[["is_feriado", "is_recesso", "is_carnaval", "is_pos_carnaval"]].sum()

# %%
# 6. Flags de feriado pro SARIMA — Varejo

df_varejo["is_feriado"] = df_varejo["dt_movimento"].dt.date.isin(datas_feriado_comum).astype(int)
df_varejo["is_recesso"] = df_varejo["dt_movimento"].dt.date.isin(datas_recesso).astype(int)
df_varejo["is_carnaval"] = df_varejo["dt_movimento"].dt.date.isin(datas_carnaval).astype(int)
df_varejo["is_pos_carnaval"] = df_varejo["dt_movimento"].dt.date.isin(datas_pos_carnaval).astype(int)

df_varejo[["is_feriado", "is_recesso", "is_carnaval", "is_pos_carnaval"]].sum()

# %%
# 7. Sazonalidade anual pro SARIMA — Prime
# O Prophet modela sazonalidade anual nativamente. O SARIMA não consegue: o termo sazonal
# dele precisaria de período 365, inviável de estimar. A saída padrão é injetar a onda anual
# como coluna exógena — pares de seno/cosseno que completam 1, 2 e 3 ciclos por ano.
# Ordem 3 (6 colunas) dá uma curva anual suave sem inchar o modelo.

EPOCA = pd.Timestamp("2020-01-01")
t_prime = (df_prime["dt_movimento"] - EPOCA).dt.days.values

df_prime["fourier_sin_1"] = np.sin(2 * np.pi * 1 * t_prime / 365.25)
df_prime["fourier_cos_1"] = np.cos(2 * np.pi * 1 * t_prime / 365.25)
df_prime["fourier_sin_2"] = np.sin(2 * np.pi * 2 * t_prime / 365.25)
df_prime["fourier_cos_2"] = np.cos(2 * np.pi * 2 * t_prime / 365.25)
df_prime["fourier_sin_3"] = np.sin(2 * np.pi * 3 * t_prime / 365.25)
df_prime["fourier_cos_3"] = np.cos(2 * np.pi * 3 * t_prime / 365.25)

df_prime.head()

# %%
# 7. Sazonalidade anual pro SARIMA — Varejo

t_varejo = (df_varejo["dt_movimento"] - EPOCA).dt.days.values

df_varejo["fourier_sin_1"] = np.sin(2 * np.pi * 1 * t_varejo / 365.25)
df_varejo["fourier_cos_1"] = np.cos(2 * np.pi * 1 * t_varejo / 365.25)
df_varejo["fourier_sin_2"] = np.sin(2 * np.pi * 2 * t_varejo / 365.25)
df_varejo["fourier_cos_2"] = np.cos(2 * np.pi * 2 * t_varejo / 365.25)
df_varejo["fourier_sin_3"] = np.sin(2 * np.pi * 3 * t_varejo / 365.25)
df_varejo["fourier_cos_3"] = np.cos(2 * np.pi * 3 * t_varejo / 365.25)

df_varejo.head()

# %%
# 8. Recortar as janelas de novo — Prime
# As janelas da célula 2 foram cortadas antes das colunas novas existirem, então
# precisam ser refeitas pra carregar as flags e os termos de Fourier.

FIM_TREINO = pd.Timestamp("2026-07-31")

df_prime_2020 = df_prime[(df_prime["dt_movimento"] >= "2020-01-01") & (df_prime["dt_movimento"] <= FIM_TREINO)]
df_prime_2023 = df_prime[(df_prime["dt_movimento"] >= "2023-01-01") & (df_prime["dt_movimento"] <= FIM_TREINO)]
df_prime_2024 = df_prime[(df_prime["dt_movimento"] >= "2024-01-01") & (df_prime["dt_movimento"] <= FIM_TREINO)]

print("linhas -> completo:", len(df_prime_2020), "| desde 2023:", len(df_prime_2023), "| desde 2024:", len(df_prime_2024))

# %%
# 8. Recortar as janelas de novo — Varejo

df_varejo_2020 = df_varejo[(df_varejo["dt_movimento"] >= "2020-01-01") & (df_varejo["dt_movimento"] <= FIM_TREINO)]
df_varejo_2023 = df_varejo[(df_varejo["dt_movimento"] >= "2023-01-01") & (df_varejo["dt_movimento"] <= FIM_TREINO)]
df_varejo_2024 = df_varejo[(df_varejo["dt_movimento"] >= "2024-01-01") & (df_varejo["dt_movimento"] <= FIM_TREINO)]

print("linhas -> completo:", len(df_varejo_2020), "| desde 2023:", len(df_varejo_2023), "| desde 2024:", len(df_varejo_2024))

# %%
# 9. Configuração do SARIMA — exógenas e o d de cada janela
# COLS_EXOG é a lista que vai no argumento `exog` do SARIMAX.
# O d de cada janela veio do ADF que rodamos na célula 3.

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
# 10. Conferência de sanidade — Prime
# Antes de jogar isso no modelo, confirmar que cada grupo se comporta como esperado.
# Esperado: feriado, recesso e carnaval ABAIXO da média; pós-carnaval ACIMA.

print("média geral:  ", round(df_prime_2020["qt_venda"].mean(), 1))
print("feriado comum:", round(df_prime_2020.loc[df_prime_2020["is_feriado"] == 1, "qt_venda"].mean(), 1))
print("recesso:      ", round(df_prime_2020.loc[df_prime_2020["is_recesso"] == 1, "qt_venda"].mean(), 1))
print("carnaval:     ", round(df_prime_2020.loc[df_prime_2020["is_carnaval"] == 1, "qt_venda"].mean(), 1))
print("pós-carnaval: ", round(df_prime_2020.loc[df_prime_2020["is_pos_carnaval"] == 1, "qt_venda"].mean(), 1))

# %%
# 10. Conferência de sanidade — Varejo

print("média geral:  ", round(df_varejo_2020["qt_venda"].mean(), 1))
print("feriado comum:", round(df_varejo_2020.loc[df_varejo_2020["is_feriado"] == 1, "qt_venda"].mean(), 1))
print("recesso:      ", round(df_varejo_2020.loc[df_varejo_2020["is_recesso"] == 1, "qt_venda"].mean(), 1))
print("carnaval:     ", round(df_varejo_2020.loc[df_varejo_2020["is_carnaval"] == 1, "qt_venda"].mean(), 1))
print("pós-carnaval: ", round(df_varejo_2020.loc[df_varejo_2020["is_pos_carnaval"] == 1, "qt_venda"].mean(), 1))
