# 1. Preparamos uma cópia do histórico apenas com as colunas que importam
df_real = df[['ano_mes', 'qt_leads']].copy()
# Criamos a nova coluna identificadora
df_real['tipo'] = 'Real'

# 2. Preparamos o DataFrame da previsão
df_pred = forecast_df.copy()
# Renomeamos a coluna projetada para ter exatamente o mesmo nome do histórico
df_pred = df_pred.rename(columns={'qt_leads_projetado': 'qt_leads'})
# Criamos a nova coluna identificadora
df_pred['tipo'] = 'Predição'

# 3. Juntamos os dois em um único DataFrame final (empilhando um no outro)
df_final = pd.concat([df_real, df_pred], ignore_index=True)

# Visualizando o resultado consolidado
print(df_final)
