import numpy as np
import pandas as pd

# 1. Parâmetros do Backtest
horizonte = 3          # Quantos meses à frente vamos prever em cada teste (ex: trimestre)
tamanho_inicial = 12   # Quantidade de meses que o modelo terá como base na primeira rodada
passos_rolling = 4     # Quantas vezes a "origem" vai andar para frente no tempo

resultados_reais = []
resultados_previstos = []

print("Iniciando o Rolling Forecast...")

# 2. Loop do Rolling Origin
for i in range(passos_rolling):
    # O ponto de corte é onde o passado termina e o futuro começa nesta iteração
    ponto_corte = tamanho_inicial + i
    
    # Valida se temos dados reais suficientes no histórico para validar esse horizonte
    if (ponto_corte + horizonte) > len(df):
        print(f"Parando na iteração {i+1}: Fim dos dados históricos disponíveis para validação.")
        break
        
    # Separa o que o modelo vai "ver" (contexto) e o que ele precisa "acertar" (gabarito)
    historico_contexto = df['qt_leads'].iloc[:ponto_corte].values
    gabarito_real = df['qt_leads'].iloc[ponto_corte : ponto_corte + horizonte].values
    
    # 3. Faz a previsão com o TimesFM
    point_forecast, _ = model.forecast(
        horizon=horizonte, 
        inputs=[historico_contexto]
    )
    previsao = point_forecast[0]
    
    # Guarda os resultados
    resultados_reais.extend(gabarito_real)
    resultados_previstos.extend(previsao)
    
    print(f"Passo {i+1} concluído.")

# 4. Cálculo das Métricas
# Convertendo para arrays do numpy para facilitar a matemática
reais_arr = np.array(resultados_reais)
previstos_arr = np.array(resultados_previstos)

# MAE (Mean Absolute Error): Erro médio absoluto em quantidade de leads
mae = np.mean(np.abs(reais_arr - previstos_arr))

# RMSE (Root Mean Squared Error): Penaliza erros maiores na projeção
rmse = np.sqrt(np.mean((reais_arr - previstos_arr)**2))

# MAPE (Mean Absolute Percentage Error): Erro médio em porcentagem
mape = np.mean(np.abs((reais_arr - previstos_arr) / reais_arr)) * 100

print("\n--- Resultados da Avaliação ---")
print(f"Total de previsões validadas: {len(reais_arr)} meses")
print(f"MAE:  {mae:.2f} leads")
print(f"RMSE: {rmse:.2f} leads")
print(f"MAPE: {mape:.2f}%")
