# %%
# 11. Ordenar cronologicamente — Prime
# O SARIMA lê a ordem das linhas como ordem do tempo, e a célula 13 calcula a
# distância entre cada linha e a anterior. As duas coisas exigem ordem cronológica.
# Os números de índice na lista de resíduos (36 = julho/2026, 1559 = junho/2020)
# mostram que o índice original não segue a data — então garanto a ordem aqui.

df_prime = df_prime.sort_values("dt_movimento").reset_index(drop=True)
df_prime[["dt_movimento", "qt_venda"]].head()

# %%
# 11. Ordenar cronologicamente — Varejo

df_varejo = df_varejo.sort_values("dt_movimento").reset_index(drop=True)
df_varejo[["dt_movimento", "qt_venda"]].head()

# %%
# 12. Remover as flags que não servem mais — Prime
# is_feriado e is_carnaval: 100% zero, porque esses dias não existem na série.
# is_pos_carnaval: fica redundante com o dias_extras da célula 13, que já dá à
# Quarta de Cinzas o valor 4 — com só 7 ocorrências, as duas colunas brigariam
# pelo mesmo efeito.
# is_recesso FICA: os dias de recesso existem na série e têm efeito próprio.
# errors="ignore" evita erro se a célula for rodada duas vezes.

df_prime = df_prime.drop(columns=["is_feriado", "is_carnaval", "is_pos_carnaval"], errors="ignore")
df_prime.columns.tolist()

# %%
# 12. Remover as flags que não servem mais — Varejo

df_varejo = df_varejo.drop(columns=["is_feriado", "is_carnaval", "is_pos_carnaval"], errors="ignore")
df_varejo.columns.tolist()

# %%
# 13. Represamento pós-feriado (dias_extras) — Prime
# Venda de dia sem expediente é implantada no próximo dia útil. Então o que importa
# é quantos dias corridos se passaram desde o último dia útil.
# Mas a segunda SEMPRE tem 3 dias (sexta -> segunda), e isso já vai ser capturado
# pelo dia da semana (célula 14). Por isso subtraio o intervalo normal de cada dia
# e fico só com o que é incomum:
#   terça comum = 0 | segunda comum = 0 | quinta após feriado na quarta = 1
#   terça após feriado na segunda = 3 | Quarta de Cinzas = 4

gap_prime = df_prime["dt_movimento"].diff().dt.days
gap_normal_prime = np.where(df_prime["dt_movimento"].dt.dayofweek == 0, 3, 1)

df_prime["dias_extras"] = (gap_prime - gap_normal_prime).fillna(0).astype(int)

# Conferência: quantos dias têm cada valor e a venda média em cada um.
# Esperado: a média SOBE conforme dias_extras sobe.
# Valor negativo = fim de semana ou data duplicada na base. Se aparecer, me avisa.
df_prime.groupby("dias_extras")["qt_venda"].agg(["count", "mean"]).round(1)

# %%
# 13. Represamento pós-feriado (dias_extras) — Varejo

gap_varejo = df_varejo["dt_movimento"].diff().dt.days
gap_normal_varejo = np.where(df_varejo["dt_movimento"].dt.dayofweek == 0, 3, 1)

df_varejo["dias_extras"] = (gap_varejo - gap_normal_varejo).fillna(0).astype(int)

df_varejo.groupby("dias_extras")["qt_venda"].agg(["count", "mean"]).round(1)

# %%
# 14. Dia da semana (dummies pro SARIMA) — Prime
# Substitui o termo sazonal s=7, que contava linhas e desalinhava com os buracos
# de dias úteis. Aqui o dia da semana vem da data real.
# Segunda é a referência (todas as colunas em 0): cada coeficiente vai dizer
# quanto aquele dia vende a menos (ou a mais) que uma segunda.
# O Prophet não precisa disso — a sazonalidade semanal dele já usa a data real.

df_prime["dow_ter"] = (df_prime["dt_movimento"].dt.dayofweek == 1).astype(int)
df_prime["dow_qua"] = (df_prime["dt_movimento"].dt.dayofweek == 2).astype(int)
df_prime["dow_qui"] = (df_prime["dt_movimento"].dt.dayofweek == 3).astype(int)
df_prime["dow_sex"] = (df_prime["dt_movimento"].dt.dayofweek == 4).astype(int)

# Conferência: contagem por dia da semana (0 = segunda ... 6 = domingo).
# Esperado: só 0 a 4. Se aparecer 5 ou 6, tem fim de semana na base.
df_prime["dt_movimento"].dt.dayofweek.value_counts().sort_index()

# %%
# 14. Dia da semana (dummies pro SARIMA) — Varejo

df_varejo["dow_ter"] = (df_varejo["dt_movimento"].dt.dayofweek == 1).astype(int)
df_varejo["dow_qua"] = (df_varejo["dt_movimento"].dt.dayofweek == 2).astype(int)
df_varejo["dow_qui"] = (df_varejo["dt_movimento"].dt.dayofweek == 3).astype(int)
df_varejo["dow_sex"] = (df_varejo["dt_movimento"].dt.dayofweek == 4).astype(int)

df_varejo["dt_movimento"].dt.dayofweek.value_counts().sort_index()

# %%
# 15. Recortar as janelas de novo — Prime
# Mesma lógica das células 2 e 8: as janelas antigas não têm as colunas novas.
# reset_index faz cada janela começar do índice 0, que é como o statsmodels prefere.

FIM_TREINO = pd.Timestamp("2026-07-31")

df_prime_2020 = df_prime[(df_prime["dt_movimento"] >= "2020-01-01") & (df_prime["dt_movimento"] <= FIM_TREINO)].reset_index(drop=True)
df_prime_2023 = df_prime[(df_prime["dt_movimento"] >= "2023-01-01") & (df_prime["dt_movimento"] <= FIM_TREINO)].reset_index(drop=True)
df_prime_2024 = df_prime[(df_prime["dt_movimento"] >= "2024-01-01") & (df_prime["dt_movimento"] <= FIM_TREINO)].reset_index(drop=True)

print("linhas -> completo:", len(df_prime_2020), "| desde 2023:", len(df_prime_2023), "| desde 2024:", len(df_prime_2024))

# %%
# 15. Recortar as janelas de novo — Varejo

df_varejo_2020 = df_varejo[(df_varejo["dt_movimento"] >= "2020-01-01") & (df_varejo["dt_movimento"] <= FIM_TREINO)].reset_index(drop=True)
df_varejo_2023 = df_varejo[(df_varejo["dt_movimento"] >= "2023-01-01") & (df_varejo["dt_movimento"] <= FIM_TREINO)].reset_index(drop=True)
df_varejo_2024 = df_varejo[(df_varejo["dt_movimento"] >= "2024-01-01") & (df_varejo["dt_movimento"] <= FIM_TREINO)].reset_index(drop=True)

print("linhas -> completo:", len(df_varejo_2020), "| desde 2023:", len(df_varejo_2023), "| desde 2024:", len(df_varejo_2024))

# %%
# 16. Checagem da sazonalidade anual — Prime
# Reaproveita os resíduos da célula 4 (precisa dela rodada): ali o Prophet
# preliminar já tirou a tendência e o padrão semanal, e NÃO tinha sazonalidade
# anual. Se existir um padrão anual, ele sobrou no resíduo.
# Divido o resíduo pelo previsto pra comparar anos com níveis diferentes na mesma
# escala: vira "% acima ou abaixo do esperado".
# Tiro 24 a 31/12 pra o efeito do recesso (que já tem flag própria) não se passar
# por sazonalidade anual.
# LEITURA: se as linhas dos anos sobem e descem JUNTAS nos mesmos meses, o padrão
# anual existe. Se cada ano vai pra um lado, não existe.

base_prime["ano"] = base_prime["ds"].dt.year
base_prime["mes"] = base_prime["ds"].dt.month
base_prime["residuo_rel"] = base_prime["residuo"] / base_prime["previsto"]

fora_recesso_prime = ~((base_prime["ds"].dt.month == 12) & (base_prime["ds"].dt.day >= 24))

sazonal_prime = base_prime[fora_recesso_prime].pivot_table(index="mes", columns="ano", values="residuo_rel", aggfunc="mean")

ax = sazonal_prime.plot(figsize=(10, 5), title="Prime — resíduo relativo médio por mês (uma linha por ano)")
sazonal_prime.mean(axis=1).plot(ax=ax, color="black", linewidth=3, label="média dos anos")
ax.axhline(0, color="gray", linewidth=1)
ax.legend()

# %%
# 16. Checagem da sazonalidade anual — Varejo

base_varejo["ano"] = base_varejo["ds"].dt.year
base_varejo["mes"] = base_varejo["ds"].dt.month
base_varejo["residuo_rel"] = base_varejo["residuo"] / base_varejo["previsto"]

fora_recesso_varejo = ~((base_varejo["ds"].dt.month == 12) & (base_varejo["ds"].dt.day >= 24))

sazonal_varejo = base_varejo[fora_recesso_varejo].pivot_table(index="mes", columns="ano", values="residuo_rel", aggfunc="mean")

ax = sazonal_varejo.plot(figsize=(10, 5), title="Varejo — resíduo relativo médio por mês (uma linha por ano)")
sazonal_varejo.mean(axis=1).plot(ax=ax, color="black", linewidth=3, label="média dos anos")
ax.axhline(0, color="gray", linewidth=1)
ax.legend()

# %%
# 17. Configuração dos modelos (substitui a célula 9)
# SARIMA: sem termo sazonal — seasonal_order fica (0, 0, 0, 0). O padrão semanal
# vem das dummies da célula 14, imunes aos buracos de dias úteis.
# Prophet: sazonalidade semanal e anual embutidas + dias_extras e is_recesso como
# regressores extras. O holidays_df sai do treino (os feriados não existem na
# série), mas continua guardado: vai montar o calendário de setembro na previsão.
# Se a célula 16 mostrar que não existe padrão anual, tiramos as 6 colunas de
# Fourier daqui e desligamos a anual do Prophet.

COLS_EXOG = [
    "dow_ter", "dow_qua", "dow_qui", "dow_sex",
    "dias_extras", "is_recesso",
    "fourier_sin_1", "fourier_cos_1",
    "fourier_sin_2", "fourier_cos_2",
    "fourier_sin_3", "fourier_cos_3",
]

REGRESSORES_PROPHET = ["dias_extras", "is_recesso"]

D_POR_JANELA = {
    ("prime", "2020"): 1, ("prime", "2023"): 1, ("prime", "2024"): 0,
    ("varejo", "2020"): 1, ("varejo", "2023"): 1, ("varejo", "2024"): 1,
}

print(len(COLS_EXOG), "exógenas no SARIMA:", COLS_EXOG)
print("regressores extras no Prophet:", REGRESSORES_PROPHET)
