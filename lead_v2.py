# %%
# Setup
# Pré-requisitos: df_prime e df_varejo já carregados, com as colunas
# `dt_movimento` (data da venda implantada) e `qt_venda` (quantidade de vendas).
# pip install pandas holidays statsmodels prophet
 
import holidays
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.stattools import adfuller
 
df_prime["dt_movimento"] = pd.to_datetime(df_prime["dt_movimento"])
df_varejo["dt_movimento"] = pd.to_datetime(df_varejo["dt_movimento"])
 
# %%
# 1. Feriados — montar o calendário nacional (fixos e móveis)
# Cobre o histórico (2020+) e o horizonte de previsão (set/2026)
 
br_holidays = holidays.Brazil(years=range(2019, 2027))
 
holidays_df = pd.DataFrame(
    [{"ds": pd.Timestamp(data), "holiday": nome} for data, nome in br_holidays.items()]
).sort_values("ds").reset_index(drop=True)
 
datas_feriado = set(holidays_df["ds"].dt.date)
# holidays_df já serve direto pro Prophet: Prophet(holidays=holidays_df)
 
holidays_df.head()
 
# %%
# 1. Feriados — aplicar no Prime (1 = feriado, 0 = dia normal)
 
df_prime["is_feriado"] = df_prime["dt_movimento"].dt.date.isin(datas_feriado).astype(int)
df_prime[["dt_movimento", "is_feriado"]].head()
 
# %%
# 1. Feriados — aplicar no Varejo
 
df_varejo["is_feriado"] = df_varejo["dt_movimento"].dt.date.isin(datas_feriado).astype(int)
df_varejo[["dt_movimento", "is_feriado"]].head()
 
# %%
# 2. Janelas de treino — Prime
# 3 janelas candidatas, todas terminando em 31/07/2026
 
FIM_TREINO = pd.Timestamp("2026-07-31")
 
df_prime_2020 = df_prime[(df_prime["dt_movimento"] >= "2020-01-01") & (df_prime["dt_movimento"] <= FIM_TREINO)]
df_prime_2023 = df_prime[(df_prime["dt_movimento"] >= "2023-01-01") & (df_prime["dt_movimento"] <= FIM_TREINO)]
df_prime_2024 = df_prime[(df_prime["dt_movimento"] >= "2024-01-01") & (df_prime["dt_movimento"] <= FIM_TREINO)]
 
print("linhas -> completo:", len(df_prime_2020), "| desde 2023:", len(df_prime_2023), "| desde 2024:", len(df_prime_2024))
 
# %%
# 2. Janelas de treino — Varejo
 
df_varejo_2020 = df_varejo[(df_varejo["dt_movimento"] >= "2020-01-01") & (df_varejo["dt_movimento"] <= FIM_TREINO)]
df_varejo_2023 = df_varejo[(df_varejo["dt_movimento"] >= "2023-01-01") & (df_varejo["dt_movimento"] <= FIM_TREINO)]
df_varejo_2024 = df_varejo[(df_varejo["dt_movimento"] >= "2024-01-01") & (df_varejo["dt_movimento"] <= FIM_TREINO)]
 
print("linhas -> completo:", len(df_varejo_2020), "| desde 2023:", len(df_varejo_2023), "| desde 2024:", len(df_varejo_2024))
 
# %%
# 3. Dickey-Fuller aumentado — Prime
# p-valor < 0.05 -> rejeita a hipótese de raiz unitária -> série estacionária
# p-valor >= 0.05 -> não estacionária -> SARIMA dessa janela provavelmente precisa de d=1
 
estatistica_prime_2020, p_valor_prime_2020, *_ = adfuller(df_prime_2020["qt_venda"].dropna(), autolag="AIC")
print(f"Prime | completo desde 2020 | estatística = {estatistica_prime_2020:.3f} | p-valor = {p_valor_prime_2020:.4f}")
 
estatistica_prime_2023, p_valor_prime_2023, *_ = adfuller(df_prime_2023["qt_venda"].dropna(), autolag="AIC")
print(f"Prime | desde 2023 | estatística = {estatistica_prime_2023:.3f} | p-valor = {p_valor_prime_2023:.4f}")
 
estatistica_prime_2024, p_valor_prime_2024, *_ = adfuller(df_prime_2024["qt_venda"].dropna(), autolag="AIC")
print(f"Prime | desde 2024 | estatística = {estatistica_prime_2024:.3f} | p-valor = {p_valor_prime_2024:.4f}")
 
# %%
# 3. Dickey-Fuller aumentado — Varejo
 
estatistica_varejo_2020, p_valor_varejo_2020, *_ = adfuller(df_varejo_2020["qt_venda"].dropna(), autolag="AIC")
print(f"Varejo | completo desde 2020 | estatística = {estatistica_varejo_2020:.3f} | p-valor = {p_valor_varejo_2020:.4f}")
 
estatistica_varejo_2023, p_valor_varejo_2023, *_ = adfuller(df_varejo_2023["qt_venda"].dropna(), autolag="AIC")
print(f"Varejo | desde 2023 | estatística = {estatistica_varejo_2023:.3f} | p-valor = {p_valor_varejo_2023:.4f}")
 
estatistica_varejo_2024, p_valor_varejo_2024, *_ = adfuller(df_varejo_2024["qt_venda"].dropna(), autolag="AIC")
print(f"Varejo | desde 2024 | estatística = {estatistica_varejo_2024:.3f} | p-valor = {p_valor_varejo_2024:.4f}")
 
# %%
# 4. Outliers pelo resíduo — Prime
# Ajuste Prophet preliminar (só sazonalidade semanal) — não é o modelo final,
# é só pra ver o que foge do padrão sem confundir "segunda-feira" com outlier
 
base_prime = df_prime[["dt_movimento", "qt_venda"]].rename(columns={"dt_movimento": "ds", "qt_venda": "y"})
 
modelo_prime = Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False)
modelo_prime.fit(base_prime)
 
previsao_prime = modelo_prime.predict(base_prime[["ds"]])
base_prime["previsto"] = previsao_prime["yhat"].values
base_prime["residuo"] = base_prime["y"] - base_prime["previsto"]
base_prime["residuo_abs"] = base_prime["residuo"].abs()
 
maiores_desvios_prime = base_prime.sort_values("residuo_abs", ascending=False).head(20)
maiores_desvios_prime[["ds", "y", "previsto", "residuo"]]
 
# %%
# 4. Outliers pelo resíduo — Varejo
 
base_varejo = df_varejo[["dt_movimento", "qt_venda"]].rename(columns={"dt_movimento": "ds", "qt_venda": "y"})
 
modelo_varejo = Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False)
modelo_varejo.fit(base_varejo)
 
previsao_varejo = modelo_varejo.predict(base_varejo[["ds"]])
base_varejo["previsto"] = previsao_varejo["yhat"].values
base_varejo["residuo"] = base_varejo["y"] - base_varejo["previsto"]
base_varejo["residuo_abs"] = base_varejo["residuo"].abs()
 
maiores_desvios_varejo = base_varejo.sort_values("residuo_abs", ascending=False).head(20)
maiores_desvios_varejo[["ds", "y", "previsto", "residuo"]]
