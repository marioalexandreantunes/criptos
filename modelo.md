# Work in progress

Plano para desenvolver um modelo de classificação binária que prevê se o preço de uma criptomoeda, como o BTC/USDT, aumentará ou diminuirá em pelo menos 5% dentro de um intervalo de 4 a 6 horas. O plano está estruturado e cobre os seguintes pontos principais:

1. **Definição do Problema e Estratégia**: Estabelece o objetivo de prever variações de preço significativas, tratando-o como um problema de classificação binária.

2. **Coleta e Pré-Processamento de Dados**: Sugere fontes de dados como APIs de exchanges (Binance, Bybit, CoinGecko) para obter dados históricos de preços e volume, além de indicadores técnicos como RSI, MACD e Bandas de Bollinger.

3. **Engenharia de Features**: Inclui exemplos de código em Python utilizando bibliotecas como `pandas` e `ta-lib` para calcular indicadores técnicos e criar novas features.

4. **Definição do Alvo (Target)**: Define a variável alvo como sendo 1 se a variação percentual futura for maior ou igual a 5% dentro do horizonte de 4 a 6 horas, caso contrário, 0.

5. **Tratamento de Classes Desbalanceadas**: Propõe técnicas como oversampling consciente de séries temporais e ajuste de pesos de classe no modelo para lidar com o possível desbalanceamento das classes.

6. **Modelagem**: Recomenda algoritmos como LightGBM, XGBoost e redes neurais temporais (LSTM/Transformers), fornecendo exemplos de código para implementação.

7. **Avaliação do Modelo**: Destaca métricas-chave como precisão, recall e AUC-ROC, além de sugerir a realização de backtesting para avaliar o desempenho do modelo em cenários de trading simulados.

8. **Otimização e Mitigação de Riscos**: Enfatiza a importância da seleção de features e tuning de hiperparâmetros, mencionando o uso de ferramentas como SHAP para interpretação do modelo e Optuna para otimização de hiperparâmetros.

9. **Implementação Prática**: Aborda considerações para monitoramento em tempo real e atualização contínua do modelo, sugerindo o uso de APIs WebSocket para receber dados em tempo real.

Este plano oferece uma abordagem abrangente para o desenvolvimento de um modelo preditivo no contexto de criptomoedas, cobrindo desde a coleta de dados até a implementação prática e monitoramento contínuo. 

---

Um projeto de código em Python que implementa um modelo de classificação binária para prever se o preço de uma criptomoeda variará pelo menos 5% dentro de uma janela de 4 a 6 horas. Este exemplo utiliza uma Rede Neural Recorrente (RNN) com Long Short-Term Memory (LSTM) para capturar padrões temporais nos dados.

**Passos principais:**

1. **Importação das bibliotecas necessárias:**

   ```python
   import numpy as np
   import pandas as pd
   from sklearn.preprocessing import StandardScaler
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import classification_report
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import LSTM, Dense, Dropout
   from tensorflow.keras.callbacks import EarlyStopping
   ```

2. **Carregamento e pré-processamento dos dados:**

   - **Carregar os dados:**

     Assumindo que tens um ficheiro CSV com os dados históricos da criptomoeda, incluindo colunas como 'timestamp', 'open', 'high', 'low', 'close', e 'volume'.

     ```python
     df = pd.read_csv('crypto_data.csv', parse_dates=['timestamp'])
     df.set_index('timestamp', inplace=True)
     ```

   - **Calcular a variação percentual futura:**

     Calcula a variação percentual do preço de fecho nos próximos 4 a 6 intervalos de tempo (assumindo que cada intervalo é de uma hora).

     ```python
     future_window = 6  # 6 horas
     df['future_return'] = df['close'].shift(-future_window) / df['close'] - 1
     ```

   - **Definir o alvo de classificação:**

     Cria uma coluna 'target' que é 1 se a variação percentual futura for maior ou igual a 5%, caso contrário, 0.

     ```python
     threshold = 0.05
     df['target'] = (df['future_return'] >= threshold).astype(int)
     ```

   - **Selecionar as features e normalizar:**

     Utiliza as colunas 'open', 'high', 'low', 'close', e 'volume' como features e normaliza-as.

     ```python
     features = ['open', 'high', 'low', 'close', 'volume']
     scaler = StandardScaler()
     df[features] = scaler.fit_transform(df[features])
     ```

   - **Remover valores nulos:**

     Remove quaisquer linhas com valores nulos resultantes do cálculo da variação futura.

     ```python
     df.dropna(inplace=True)
     ```

3. **Preparação dos dados para o modelo LSTM:**

   - **Definir a janela de observação:**

     Define quantas horas de dados anteriores serão usadas para prever o movimento futuro.

     ```python
     observation_window = 24  # 24 horas
     ```

   - **Criar sequências de dados:**

     Cria sequências de dados para alimentar o modelo LSTM.

     ```python
     def create_sequences(data, target, window):
         X, y = [], []
         for i in range(len(data) - window):
             X.append(data[i:i + window])
             y.append(target[i + window])
         return np.array(X), np.array(y)

     X, y = create_sequences(df[features].values, df['target'].values, observation_window)
     ```

   - **Dividir os dados em conjuntos de treino e teste:**

     Divide os dados em conjuntos de treino e teste.

     ```python
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
     ```

4. **Construção e treino do modelo LSTM:**

   - **Definir o modelo:**

     Cria um modelo sequencial com camadas LSTM e Dropout para evitar overfitting.

     ```python
     model = Sequential()
     model.add(LSTM(50, return_sequences=True, input_shape=(observation_window, len(features))))
     model.add(Dropout(0.2))
     model.add(LSTM(50))
     model.add(Dropout(0.2))
     model.add(Dense(1, activation='sigmoid'))
     ```

   - **Compilar o modelo:**

     Compila o modelo com o otimizador Adam e a função de perda binária.

     ```python
     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
     ```

   - **Treinar o modelo:**

     Treina o modelo com early stopping para evitar overfitting.

     ```python
     early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
     history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, callbacks=[early_stopping])
     ```

5. **Avaliação do modelo:**

   - **Avaliar o desempenho no conjunto de teste:**

     Avalia o modelo no conjunto de teste e imprime um relatório de classificação.

     ```python
     y_pred = (model.predict(X_test) > 0.5).astype(int)
     print(classification_report(y_test, y_pred))
     ```

**Considerações finais:**

- **Dados:** Certifica-te de que tens dados suficientes para treinar o modelo, cobrindo diferentes condições de mercado.

- **Features adicionais:** Considera adicionar indicadores técnicos ou dados de sentimento para melhorar o desempenho do modelo.

- **Validação:** Utiliza técnicas de validação cruzada e backtesting para avaliar a robustez do modelo.

- **Riscos:** Lembra-te de que a negociação de criptomoedas envolve riscos significativos. Testa exaustivamente qualquer modelo antes de o utilizar em cenários de negociação real.

Este é um exemplo básico para te ajudar a começar. Dependendo das especificidades do teu projeto, poderás precisar de ajustar e expandir este código. 
