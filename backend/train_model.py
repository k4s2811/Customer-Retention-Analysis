import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib, os

# Load dataset
df = pd.read_csv("./dataset/data.csv").dropna()

# Encode categorical values
for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Features & target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Ensure model folder exists
os.makedirs("model", exist_ok=True)

# Save model
joblib.dump(rf_model, "model/churn_model.pkl")
print("✅ Model retrained and saved at backend/model/churn_model.pkl")
