import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r"C:\Users\KML\Downloads\diabetes (1).csv")
print(df)

age_min=int(input('enter the minimum age :'))
age_max=int(input('enter the maximum age :'))

filter_df=df[(df['Age']>=age_min)&(df['Age']<=age_max)]
print(filter_df)

x=filter_df.drop('Outcome',axis=1)
y=filter_df['Outcome']
print(x)
print(y)

x_train, x_test, y_train, y_test =train_test_split(x, y, test_size=0.2 ,random_state=42)

model=RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

y_pred=model.predict(x_test)

print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))
feature_imp = pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.title(f"Feature Importance (Age {age_min}-{age_max})")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
