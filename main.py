from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle

# Данные: площадь дома (кв. м.) и цена (тыс. долларов)
X = np.array([[30], [40], [50], [60], [70], [80], [90], [100]])
y = np.array([100, 150, 200, 250, 300, 350, 400, 450])

# Создаем и обучаем модель
model = LinearRegression()
model.fit(X, y)

# Сохраняем модель в файл
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

# Загружаем модель (на случай, если она нужна будет отдельно)
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Инициализируем FastAPI
app = FastAPI()

# Определяем схему запроса с помощью Pydantic
class PredictionRequest(BaseModel):
    area: float

# Определяем эндпоинт для предсказаний
"/predict"
def predict_price(request: PredictionRequest):
    area = np.array([[request.area]])
    prediction = model.predict(area)
    return {"predicted_price": prediction[0]}

# Для запуска приложения, команду:
# uvicorn app:app —reload