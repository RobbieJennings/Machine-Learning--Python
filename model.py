import pandas as pd
from math import sqrt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from preprocessing import quantify_data, remove_outliers

# Set to True to enable local testing
test = True

# Load the filenames
train_input = "tcd ml 2019-20 income prediction training (with labels).csv"
test_input = "tcd ml 2019-20 income prediction test (without labels).csv"

# Load the datasets
train_data = pd.read_csv(train_input, header=0)
test_data = pd.read_csv(test_input, header=0)

# Rename Income column to match test data
train_data = train_data.rename(columns={'Income in EUR': 'Income'})

# Perform local testing using only train data
if(test):
    train_data, test_data = train_test_split(train_data, test_size=0.2)
    incomes = test_data["Income"]

# Remove outliers from training data
train_data = remove_outliers(train_data)

# Quantify the datasets
test_data = quantify_data(test_data, train_data)
train_data = quantify_data(train_data, train_data)

# Split the data into training/testing sets
train_x = train_data.drop(["Income", "Instance"], axis=1)
test_x = test_data.drop(["Income", "Instance"], axis=1)

# Split the targets into training/testing sets
train_y = train_data["Income"]
test_y = test_data["Income"]

# Fill missing test data
for column in train_x.columns:
    if column not in test_x.columns:
        test_x[column] = [0] * len(test_y)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(train_x, train_y)

# Make predictions using the testing set
pred_y = regr.predict(test_x)

# Load predictions into DataFrame
if not test:
    predictions = {"Instance": range(111994, 185224), "Income": pred_y}
    predictions = pd.DataFrame(predictions)

# Save predictions to file
if not test:
    predictions = predictions.to_csv(index=False)
    file = open("tcd ml 2019-20 income prediction submission file.csv", "w+")
    file.write(predictions)
    file.close()

# Check results
if test:
    incomes = incomes.tolist()
    results = pred_y.tolist()
    rms = sqrt(mean_squared_error(incomes, results))
    print(rms)
