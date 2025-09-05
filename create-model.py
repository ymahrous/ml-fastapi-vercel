import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

cars_data = pd.read_csv('cars.csv')
X = cars_data.drop(columns=['vehicle_type']).values
y = cars_data['vehicle_type']

model = DecisionTreeClassifier()
model.fit(X, y)

predictions = model.predict([[22,1]])
print(predictions[0])

# joblib.dump(model, 'car-recommender.joblib')