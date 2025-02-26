import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("logic_depth.csv")

# Define inputs (X) and output (y)
X = df[["Num_Gates", "Fan_In"]]
y = df["Combinational_Depth"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple ML model
model = LinearRegression()
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)
print("Predicted Logic Depths:", predictions)
