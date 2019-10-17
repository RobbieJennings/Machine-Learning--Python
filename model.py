import pandas as pd
from math import sqrt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from preprocessing import remove_outliers
from preprocessing import quantify_data
from preprocessing import polynomialize_data
from preprocessing import standardize_data
from preprocessing import normalize_data

# Environment Variables
test = True
outliers = False
polynomialize = True
standardize = True
normalize = False

# Declare columns to drop from training data
unnecessary = ["Instance"]

# Create linear regression object
regr = MLPRegressor(hidden_layer_sizes=(100,),
                    activation='relu',
                    solver='adam',
                    learning_rate='constant',
                    max_iter=1000,
                    learning_rate_init=0.001,
                    alpha=0.0001)

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

# Drop unnecessary columns
train_data = train_data.drop(unnecessary, axis=1)
test_data = test_data.drop(unnecessary, axis=1)

# Remove outliers from training data
if(outliers):
    train_data = remove_outliers(train_data)

# Quantify the datasets
train_length = train_data.shape[0]
data = train_data.append(test_data)
data = quantify_data(data)
train_data = data[:train_length]
test_data = data[train_length:]

# Split the data into training/testing sets
train_x = train_data.drop(["Income"], axis=1)
test_x = test_data.drop(["Income"], axis=1)

# Split the targets into training/testing sets
train_y = train_data["Income"]
test_y = test_data["Income"]

# Polynomialize the datasets
if(polynomialize):
    train_x = polynomialize_data(train_x)
    test_x = polynomialize_data(test_x)

# Standardize the datasets
if(standardize):
    train_x = standardize_data(train_x)
    test_x = standardize_data(test_x)

# Normalize the datasets
if(normalize):
    train_x = normalize_data(train_x)
    test_x = normalize_data(test_x)

# Train the model using the training sets
# and make predictions using testing sets
regr.fit(train_x, train_y)
pred_y = regr.predict(test_x)

# Save results to file
if not test:
    predictions = {"Instance": range(111994, 185224), "Income": pred_y}
    predictions = pd.DataFrame(predictions)
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
