import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load the dataset
df = pd.read_csv(r"C:\Users\KML\Downloads\diabetes.csv")

# Features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = classifier.predict(X_test_scaled)
print("Initial Model Performance")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

# Best model
best_model = grid_search.best_estimator_
print("\nBest Parameters:\n", grid_search.best_params_)

# Evaluate the best model
y_pred_best = best_model.predict(X_test_scaled)
print("\nBest Model Performance")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))
print("\nClassification Report:\n", classification_report(y_test, y_pred_best))

# Save the model and scaler
joblib.dump(best_model, 'diabetes_classifier.pkl')
joblib.dump(scaler, 'diabetes_scaler.pkl')

# Load and make prediction on new data
loaded_model = joblib.load('diabetes_classifier.pkl')
loaded_scaler = joblib.load('diabetes_scaler.pkl')

sample_data = X_test.iloc[:5]
sample_scaled = loaded_scaler.transform(sample_data)
predictions = loaded_model.predict(sample_scaled)

print("\nPredictions on New Samples:\n", predictions)
print("Actual Labels:\n", y_test.iloc[:5].values)
