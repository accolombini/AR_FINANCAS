import pandas as pd
import matplotlib.pyplot as plt

# Caminho do arquivo gerado
caminho_csv = "/Users/accol/Library/Mobile Documents/com~apple~CloudDocs/PROJETOS/ar_financas/BackPython/DADOS/VALE3.SA_prophet_previsao.csv"

# Carregar os dados de previsão
dados_previsao = pd.read_csv(caminho_csv)

# Limitar os dados ao horizonte de 6 meses (126 dias úteis)
dados_previsao['ds'] = pd.to_datetime(dados_previsao['ds'])
dados_6_meses = dados_previsao.head(126)

# Plotar os resultados
plt.figure(figsize=(14, 7))
plt.plot(dados_6_meses['ds'], dados_6_meses['yhat'],
         label='Previsão (yhat)', color='blue')
plt.fill_between(dados_6_meses['ds'], dados_6_meses['yhat_lower'], dados_6_meses['yhat_upper'],
                 color='blue', alpha=0.2, label='Intervalo de Confiança (95%)')
plt.title('Previsão com Prophet - Horizonte de 6 Meses')
plt.xlabel('Data')
plt.ylabel('Preço Estimado')
plt.legend()
plt.grid()
plt.show()
