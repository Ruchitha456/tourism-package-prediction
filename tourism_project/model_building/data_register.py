
import os
import pandas as pd

# Define the dataset path
DATA_PATH = "tourism_project/data/tourism.csv"

# Expected columns based on the dataset description
EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome"
]

# Check whether the dataset exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Validate the expected columns
missing_columns = [
    column for column in EXPECTED_COLUMNS
    if column not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing expected columns: {missing_columns}")

# Print the dataset summary
print("Dataset registration and validation successful!")
print("\nDataset shape:", df.shape)
print("\nColumn validation: Passed")
print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nFirst five rows:")
print(df.head())

print("\nTarget variable distribution:")
print(df["ProdTaken"].value_counts())
