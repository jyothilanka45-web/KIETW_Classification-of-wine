import pickle
from pathlib import Path

# Load saved model and scaler
model_path = Path.cwd() / "model_rf.pkl"
scaler_path = Path.cwd() / "scaler.pkl"

# Save the objects already trained in this notebook
if not model_path.exists():
    with open(model_path, "wb") as f:
        pickle.dump(model_rf, f)

if not scaler_path.exists():
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

# Load the saved model
with open(model_path, "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# List of feature names in correct order
feature_names = [
    'fixed acidity', 'volatile acidity', 'citric acid', 
    'residual sugar','chlorides', 'free sulfur dioxide', 
    'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

# Collect input values from user
print("Enter the following details for prediction:")

user_values = []
for feature in feature_names:
    val = float(input(f"Enter {feature}: "))
    user_values.append(val)

# Convert to array and reshape
input_data = np.array(user_values).reshape(1, -1)

# Scale using the fitted scaler
scaled_data = scaler.transform(input_data)

# Predict
prediction = model.predict(scaled_data)

print(f"\nPredicted Wine Quality: {prediction[0]}")