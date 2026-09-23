# %%
# 18. Novo corte: treino só com dado completo (até 30/06/2026)
# Julho e agosto de 2026 estão incompletos — falta um arquivo de origem que não
# chega com o tempo, então parte das vendas desses meses não é identificável.
# Como o corte é na PONTA, o dias_extras e as dummies das células 13 e 14 seguem
# válidos: cada linha depende só dela e da anterior. Não precisa re-rodar 11 a 15.
#
# ATENÇÃO: isto substitui o FIM_TREINO da célula 15. As células 18 a 22 da versão
# anterior (exploração de agosto) podem ser apagadas.

FIM_TREINO = pd.Timestamp("2026-06-30")

df_prime_ok = df_prime[df_prime["dt_movimento"] <= FIM_TREINO].reset_index(drop=True)
df_varejo_ok = df_varejo[df_varejo["dt_movimento"] <= FIM_TREINO].reset_index(drop=True)

print("Prime  — última data:", df_prime_ok["dt_movimento"].max(), "| linhas:", len(df_prime_ok))
print("Varejo — última data:", df_varejo_ok["dt_movimento"].max(), "| linhas:", len(df_varejo_ok))

# %%
# 18. Recortar as janelas sobre o dado completo — Prime

df_prime_2020 = df_prime_ok[df_prime_ok["dt_movimento"] >= "2020-01-01"].reset_index(drop=True)
df_prime_2023 = df_prime_ok[df_prime_ok["dt_movimento"] >= "2023-01-01"].reset_index(drop=True)
df_prime_2024 = df_prime_ok[df_prime_ok["dt_movimento"] >= "2024-01-01"].reset_index(drop=True)

print("linhas -> completo:", len(df_prime_2020), "| desde 2023:", len(df_prime_2023), "| desde 2024:", len(df_prime_2024))

# %%
# 18. Recortar as janelas sobre o dado completo — Varejo

df_varejo_2020 = df_varejo_ok[df_varejo_ok["dt_movimento"] >= "2020-01-01"].reset_index(drop=True)
df_varejo_2023 = df_varejo_ok[df_varejo_ok["dt_movimento"] >= "2023-01-01"].reset_index(drop=True)
df_varejo_2024 = df_varejo_ok[df_varejo_ok["dt_movimento"] >= "2024-01-01"].reset_index(drop=True)

print("linhas -> completo:", len(df_varejo_2020), "| desde 2023:", len(df_varejo_2023), "| desde 2024:", len(df_varejo_2024))

# %%
# 19. ADF de novo, agora sem os meses furados — Prime
# O resultado PODE mudar: as janelas encurtaram, e antes o teste estava vendo dois
# meses artificialmente baixos no fim da série — o que empurra na direção de
# "não estacionária". Atenção à janela de 2024, que deu 0.0285, rente aos 0.05.

estatistica_prime_2020, p_valor_prime_2020, *_ = adfuller(df_prime_2020["qt_venda"].dropna(), autolag="AIC")
print(f"Prime | completo   | p-valor = {p_valor_prime_2020:.4f} | antes: 0.3189")

estatistica_prime_2023, p_valor_prime_2023, *_ = adfuller(df_prime_2023["qt_venda"].dropna(), autolag="AIC")
print(f"Prime | desde 2023 | p-valor = {p_valor_prime_2023:.4f} | antes: 0.1429")

estatistica_prime_2024, p_valor_prime_2024, *_ = adfuller(df_prime_2024["qt_venda"].dropna(), autolag="AIC")
print(f"Prime | desde 2024 | p-valor = {p_valor_prime_2024:.4f} | antes: 0.0285  <-- borderline")

# %%
# 19. ADF de novo — Varejo

estatistica_varejo_2020, p_valor_varejo_2020, *_ = adfuller(df_varejo_2020["qt_venda"].dropna(), autolag="AIC")
print(f"Varejo | completo   | p-valor = {p_valor_varejo_2020:.4f} | antes: 0.4636")

estatistica_varejo_2023, p_valor_varejo_2023, *_ = adfuller(df_varejo_2023["qt_venda"].dropna(), autolag="AIC")
print(f"Varejo | desde 2023 | p-valor = {p_valor_varejo_2023:.4f} | antes: 0.6771")

estatistica_varejo_2024, p_valor_varejo_2024, *_ = adfuller(df_varejo_2024["qt_venda"].dropna(), autolag="AIC")
print(f"Varejo | desde 2024 | p-valor = {p_valor_varejo_2024:.4f} | antes: 0.3508")

# %%
# 20. Refazer a lista de outliers sobre o dado limpo — Prime
# A lista da célula 4 foi gerada com julho e agosto dentro. Dois problemas:
#   1. as datas de julho/2026 entupiram o top-20 com um erro já explicado;
#   2. pior — dois meses artificialmente baixos no fim puxaram a TENDÊNCIA do
#      Prophet preliminar para baixo, o que distorce os resíduos vizinhos também.
# Refazendo sem eles, desvios reais que estavam escondidos devem aparecer.

base_prime = df_prime_ok[["dt_movimento", "qt_venda"]].rename(columns={"dt_movimento": "ds", "qt_venda": "y"})

modelo_prime = Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False)
modelo_prime.fit(base_prime)

previsao_prime = modelo_prime.predict(base_prime[["ds"]])
base_prime["previsto"] = previsao_prime["yhat"].values
base_prime["residuo"] = base_prime["y"] - base_prime["previsto"]
base_prime["residuo_abs"] = base_prime["residuo"].abs()

maiores_desvios_prime = base_prime.sort_values("residuo_abs", ascending=False).head(10)
maiores_desvios_prime[["ds", "y", "previsto", "residuo"]].round(1)

# %%
# 20. Refazer a lista de outliers — Varejo

base_varejo = df_varejo_ok[["dt_movimento", "qt_venda"]].rename(columns={"dt_movimento": "ds", "qt_venda": "y"})

modelo_varejo = Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False)
modelo_varejo.fit(base_varejo)

previsao_varejo = modelo_varejo.predict(base_varejo[["ds"]])
base_varejo["previsto"] = previsao_varejo["yhat"].values
base_varejo["residuo"] = base_varejo["y"] - base_varejo["previsto"]
base_varejo["residuo_abs"] = base_varejo["residuo"].abs()

maiores_desvios_varejo = base_varejo.sort_values("residuo_abs", ascending=False).head(10)
maiores_desvios_varejo[["ds", "y", "previsto", "residuo"]].round(1)
