# Work in progress

Vamos construir um modelo de classificação binária para prever variações de preço em crypto, seguindo um plano estruturado e adaptado às particularidades do mercado. Aqui está o passo a passo detalhado:

---

### **1. Definição do Problema e Estratégia**
- **Objetivo**: Prever se o preço de uma crypto (ex: BTC/USDT) aumentará ou diminuirá **5% em 4 a 6 horas**.
- **Classificação Binária**:
  - Classe 1 (Positiva): Preço varia ≥ 5% dentro de 4-6 horas.
  - Classe 0 (Negativa): Preço varia < 5% no mesmo período.
- **Cuidado**: Mercados de crypto são altamente estocásticos – mesmo um modelo bom terá acurácia limitada. Foque em **precisão/recall da classe positiva**.

---

### **2. Coleta e Pré-Processamento de Dados**
#### **Fontes de Dados**:
- **APIs de Exchanges**: Binance, Bybit, CoinGecko (dados históricos de preço, volume, ordem book).
- **Indicadores Técnicos**:
  - RSI, MACD, Bollinger Bands, Volume, Volatilidade (desvio padrão dos retornos).
  - Médias móveis (7, 20, 50 períodos).
- **Dados On-Chain** (opcional): Número de transações, hash rate (para Bitcoin), active addresses.
- **Janelas Temporais**:
  - Use candles de **15 minutos** ou **1 hora** para capturar padrões de curto prazo.

#### **Engenharia de Features**:
```python
import pandas as pd
import talib

# Exemplo: Criar features para cada candle
df['RSI'] = talib.RSI(df['close'], timeperiod=14)
df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'])
df['volatility'] = df['close'].rolling(24).std()  # Volatilidade de 24 períodos
```

#### **Definição do Target**:
```python
# Calcular variação percentual nas próximas 4-6 horas
horizon = 6  # Janela de 6 horas
df['future_price'] = df['close'].shift(-horizon)
df['target'] = (df['future_price'].pct_change(horizon) >= 0.05).astype(int)  # 5% de alta
df.dropna(inplace=True)  # Remover NaN
```

---

### **3. Tratamento de Classes Desbalanceadas**
Se a classe 1 (variação ≥5%) for rara (<5% dos dados):
- **Técnicas para Séries Temporais**:
  - **Time-Series Aware Oversampling**: Replicar eventos raros com pequenas variações de tempo (ex: copiar o evento e deslocar ±1 candle).
  - **Pesos de Classe no Modelo**: Penalizar erros na classe minoritária.
  ```python
  from sklearn.utils.class_weight import compute_class_weight

  class_weights = compute_class_weight('balanced', classes=[0, 1], y=df['target'])
  model = XGBClassifier(scale_pos_weight=class_weights[1]/class_weights[0])
  ```
- **Evite SMOTE Tradicional**: Gera dados sintéticos que quebram a estrutura temporal.

---

### **4. Modelagem**
#### **Algoritmos Recomendados**:
- **LightGBM/XGBoost**: Lidam bem com desbalanceamento e features não-lineares.
- **Redes Neurais Temporais (LSTM/Transformers)**: Úteis se houver dados suficientes (>100k amostras).
- **Ensemble Híbrido**: Combine modelos de série temporal (ex: Prophet) com LightGBM.

#### **Exemplo com LightGBM**:
```python
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

# Separar features e target
X = df.drop(['target', 'future_price', 'close'], axis=1)
y = df['target']

# Validação Cruzada Temporal
tscv = TimeSeriesSplit(n_splits=3)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Treinar modelo
    model = lgb.LGBMClassifier(
        class_weight='balanced',
        objective='binary',
        metric='auc'
    )
    model.fit(X_train, y_train)
    
    # Avaliar
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
```

---

### **5. Avaliação do Modelo**
#### **Métricas-Chave**:
- **Precision (Classe 1)**: % de previsões corretas de variação ≥5%.
- **Recall (Classe 1)**: % de eventos reais de 5% capturados pelo modelo.
- **AUC-ROC**: Mede a capacidade de distinguir entre as classes.
- **Matriz de Confusão**: Analise falsos positivos/negativos.

#### **Backtesting**:
- **Simulação de Trading**: Aplique o modelo em dados históricos com regras de entrada/saída.
  - Exemplo: Comprar quando o modelo prevê classe 1 e vender após 5% de lucro ou 6 horas.
- **Métricas Financeiras**:
  - **Sharpe Ratio**: Risco-retorno.
  - **Maximum Drawdown**: Perda máxima acumulada.
  - **Profit Factor**: (Ganhos Totais) / (Perdas Totais).

---

### **6. Otimização e Mitigação de Riscos**
- **Feature Selection**:
  - Remova features correlacionadas com `df.corr()`.
  - Use SHAP para interpretar importância de features:
  ```python
  import shap
  explainer = shap.TreeExplainer(model)
  shap_values = explainer.shap_values(X_test)
  shap.summary_plot(shap_values, X_test)
  ```
- **Hyperparameter Tuning**:
  ```python
  from optuna import create_study

  study = create_study(direction='maximize')
  study.optimize(objective_function, n_trials=50)  # Otimizar AUC-ROC
  ```

---

### **7. Implementação Prática**
- **Monitoramento em Tempo Real**:
  - Atualize o modelo semanalmente com novos dados.
  - Use APIs WebSocket para receber dados em tempo real (ex: Binance WebSocket).
- **Gerenciamento de Risco**:
  - Defina stop-loss (ex: -3%) e take-profit (5%).
  - Limite o tamanho das posições com base na confiança do modelo.

---

### **8. Desafios Comuns e Soluções**
| **Desafio**                     | **Solução**                                      |
|---------------------------------|--------------------------------------------------|
| Overfitting em dados sintéticos | Use validação temporal e evite SMOTE             |
| Baixo recall da classe 1        | Aumente o peso da classe 1 ou use undersampling  |
| Alta volatilidade do mercado    | Adicione features de volatilidade recente        |
| Atraso na execução de ordens    | Teste o modelo em dados com slippage realista    |

---

### **Exemplo de Código para Backtesting**
```python
# Simulação simplificada de trading
capital = 10000
position = 0

for i in range(len(predictions)):
    if predictions[i] == 1 and position == 0:
        # Comprar no preço atual
        position = capital / df['close'].iloc[i]
        capital = 0
    elif position > 0:
        # Vender após 5% de lucro ou 6 horas
        current_price = df['close'].iloc[i]
        if current_price >= entry_price * 1.05 or i >= entry_index + 6:
            capital = position * current_price
            position = 0

print(f"Capital final: ${capital:.2f}")
```

---

### **Conclusão**
Prever variações de 5% em crypto é um problema difícil, mas viável com:
- **Features robustas** (volatilidade, volume, indicadores técnicos).
- **Modelos adaptados a séries temporais** (LightGBM com validação cronológica).
- **Backtesting rigoroso** com métricas financeiras.

Comece com janelas menores (ex: 1-2 horas) para validar a ideia antes de escalar. Compartilhe mais detalhes do seu dataset para ajustarmos o modelo! 🚀
