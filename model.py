import pandas as pd
from math import sqrt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from preprocessing import remove_outliers
from preprocessing import quantify_data
from preprocessing import standardize_data
from preprocessing import normalize_data

# Set to True to enable local testing
test = False

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
    input = test_data

# Drop unnecessary columns
unnecessary = ["Instance"]
train_data = train_data.drop(unnecessary, axis=1)
test_data = test_data.drop(unnecessary, axis=1)

# Remove outliers from training data
train_data = remove_outliers(train_data)

# Quantify the datasets
length = train_data.shape[0]
data = train_data.append(test_data)
data = quantify_data(data)
train_data = data[:length]
test_data = data[length:]

# Split the data into training/testing sets
train_x = train_data.drop(["Income"], axis=1)
test_x = test_data.drop(["Income"], axis=1)

# Split the targets into training/testing sets
train_y = train_data["Income"]
test_y = test_data["Income"]

# Fill missing test data
for column in train_x.columns:
    if column not in test_x.columns:
        test_x[column] = [0] * len(test_y)

# Standardize the datasets
# train_x = standardize_data(train_x)
# test_x = standardize_data(test_x)

# Normalize the datasets
# train_x = normalize_data(train_x)
# test_x = normalize_data(test_x)

# Create linear regression object
regr = linear_model.Lasso(alpha=0.1)

# Train the model using the training sets
regr.fit(train_x, train_y)

# Make predictions using the testing set
pred_y = regr.predict(test_x)

# Make submission file
if not test:
    # Load predictions into DataFrame
    predictions = {"Instance": range(111994, 185224), "Income": pred_y}
    predictions = pd.DataFrame(predictions)

    # Save predictions to file
    predictions = predictions.to_csv(index=False)
    file = open("tcd ml 2019-20 income prediction submission file.csv", "w+")
    file.write(predictions)
    file.close()

# Check results
if test:
    incomes = test_y.tolist()
    results = pred_y.tolist()
    rms = sqrt(mean_squared_error(incomes, results))
    print(rms)
