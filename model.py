import pandas as pd
from sklearn import linear_model

from preprocessing import quantify_data


# Load the filenames
train_input = "tcd ml 2019-20 income prediction training (with labels).csv"
test_input = "tcd ml 2019-20 income prediction test (without labels).csv"

# Load the datasets
train_data = pd.read_csv(train_input, header=0)
test_data = pd.read_csv(test_input, header=0)
train_data = train_data.rename(columns={'Income in EUR': 'Income'})

# Quantify the datasets
train_data = quantify_data(train_data, train_data)
test_data = quantify_data(test_data, train_data)

# Split the data into training/testing sets
train_x = train_data[["Year of Record", "Gender", "Age",
                      "Country", "Size of City", "Profession",
                      "University Degree", "Wears Glasses",
                      "Hair Color", "Body Height [cm]"]]
test_x = test_data[["Year of Record", "Gender", "Age",
                    "Country", "Size of City", "Profession",
                    "University Degree", "Wears Glasses",
                    "Hair Color", "Body Height [cm]"]]

# Split the targets into training/testing sets
train_y = train_data["Income"]
test_y = test_data["Income"]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(train_x, train_y)

# Make predictions using the testing set
pred_y = regr.predict(test_x)

# Load predictions into DataFrame
predictions = {"Instance": range(
    train_data["Instance"].size + 1,
    train_data["Instance"].size + 1 + test_data["Instance"].size),
    "Income": pred_y}
predictions = pd.DataFrame(predictions)

# Save predictions to file
predictions = predictions.to_csv(index=False)
file = open("tcd ml 2019-20 income prediction submission file.csv", "w+")
file.write(predictions)
file.close()
