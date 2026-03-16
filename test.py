# ... [Seu código anterior do model.compile fica exatamente igual] ...

print("Gerando o forecast...")

# 1. Extraímos os valores da coluna de leads como uma lista de arrays numpy
valores_historicos = [df['qt_leads'].values]

# 2. Usamos o método nativo 'forecast' da versão 2.5
# Ele retorna dois objetos: a previsão exata (point) e os intervalos (quantiles)
point_forecast, quantile_forecast = model.forecast(
    horizon=6, 
    inputs=valores_historicos
)

# 3. O resultado é uma matriz. Extraímos a primeira (e única) série do nosso teste
previsao_leads = point_forecast[0]

# 4. Descobrimos qual foi o último mês do histórico e criamos as datas futuras
ultima_data = df['ano_mes'].max()
datas_futuras = pd.date_range(
    start=ultima_data + pd.DateOffset(months=1), 
    periods=6, 
    freq='MS' # Frequência mensal (MS = Month Start)
)

# 5. Montamos o DataFrame final limpo e pronto!
forecast_df = pd.DataFrame({
    'ano_mes': datas_futuras,
    'qt_leads_projetado': previsao_leads
})

print(forecast_df)
