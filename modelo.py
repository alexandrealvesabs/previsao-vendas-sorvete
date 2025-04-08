import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

# Carregar os dados
data = pd.read_csv('vendas_temperatura.csv')

# Preparar os dados
X = data[['temperatura']]
y = data['vendas']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Registrar o modelo com MLflow
mlflow.start_run()
mlflow.log_param("model_type", "Linear Regression")
mlflow.log_metric("mse", mse)
mlflow.log_metric("r2", r2)
mlflow.sklearn.log_model(model, "model")
mlflow.end_run()

print(f'MSE: {mse}, R²: {r2}')
