import pandas as pd

# Load your dataset
df = pd.read_csv(r"C:\Users\KML\Downloads\diabetes.csv")

# Show the number of null values in each column
print(df.isnull().sum())

