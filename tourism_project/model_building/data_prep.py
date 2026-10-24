
import os
import pandas as pd 
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [ 
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome"
]

#  Load the registered dataset
if not os.path.exists(DATA_PATH):
   raise FileNotFoundError(f"Dataset not found: {DATA_PATH}") 

df=pd.read_csv(DATA_PATH) 

# Validate the dataset schema
missing_columns =[
    column for column in EXPECTED_COLUMNS if column not in df.columns
]
if missing_columns:
   raise ValueError(f"Required columns are missing: {missing_columns}")
if df.empty:
   raise ValueError("The dataset is empty") 
print("Dataset loaded successfully:",df.shape) 

# Remove duplicate records and unnecessary identifier columns
df = df.drop(columns=["Unnamed: 0","CustomerID"],errors="ignore")
df = df.drop_duplicates()

# Standardize inconsistent categorical labels
df["Gender"] = df["Gender"].replace({"Fe Male":"Female"})
df["Occupation"] = df["Occupation"].replace({"Free Lancer":"Freelancer"})

# Validate data quality before splitting
if df.isnull().any().any():
    raise ValueError(
        "Missing values detected.Resolve them before model training."
    )

if not set(df["ProdTaken"].unique()).issubset({0, 1}):
    raise ValueError("ProdTaken must contain only binary labels 0 and 1.")

print("Data cleaning and validation completed.")
print("Cleaned dataset shape:",df.shape)
print("Duplicate records remaining:",df.duplicated().sum())

# Separate model features and the prediction target
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Split the data while preserving the target class distribution
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=42,stratify=y)

# Save the prepared datasets for the training stage
output_files = {
    "Xtrain.csv": X_train,
    "Xtest.csv": X_test,
    "ytrain.csv": y_train.to_frame(name="ProdTaken"),
    "ytest.csv": y_test.to_frame(name="ProdTaken")
}
for filename, data in output_files.items():
    data.to_csv(filename,index=False)

print("\nData preparation completed successfully.")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target distribution:")
print(y_train.value_counts(normalize=True).round(3))
print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).round(3))
print("\nPrepared files saved:")
for filename in output_files:
    print(f"- {filename}")
