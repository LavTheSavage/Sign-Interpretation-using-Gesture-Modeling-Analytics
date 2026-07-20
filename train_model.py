import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

print("Loading dataset...")
df = pd.read_csv('gesture_data.csv', header=None)


y = df.iloc[:, 0]      
X = df.iloc[:, 1:]       

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Training Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Training Complete!")
print(f"Accuracy Score: {accuracy * 100:.2f}%")


with open('sign_language_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Saved trained model to 'sign_language_model.pkl'")