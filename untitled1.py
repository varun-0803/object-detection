import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv(r"C:\Users\KML\Downloads\diabetes (1).csv")

# User input for age range
age_min = int(input('Enter the minimum age: '))
age_max = int(input('Enter the maximum age: '))

# Filter dataset based on age range
age_filtered_df = df[(df['Age'] >= age_min) & (df['Age'] <= age_max)]

20

age_filtered_df.loc[:, 'filtered_age'] = pd.cut(age_filtered_df['Age'], bins=[10,20,30,40,50,60,70,80])

# Drop non-numeric column before ML
X = age_filtered_df.drop(['Outcome', 'filtered_age'], axis=1)
y = age_filtered_df['Outcome']


# Show age distribution as table
age_counts = age_filtered_df['Age'].value_counts().sort_index()
age_count_df = pd.DataFrame({'Age': age_counts.index, 'Count': age_counts.values})
print("\nCount of Each Age in the Selected Range:")
print(age_count_df)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Compare actual vs predicted
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

print("\nFiltered Data:")
print(age_filtered_df.head())

print("\nPrediction Results:")
print(results_df)

# Classification report
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

print("\nClassification Report:")
print(report_df)
