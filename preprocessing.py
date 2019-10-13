import numpy as np
import pandas as pd
from scipy import stats


# Replace unkowns in Year of Record with mean and convert to age of record
def quantify_year(data, train_data):
    data["Year of Record"] = data["Year of Record"].fillna(
        int(data["Year of Record"].mean()))
    data["Year of Record"] = (data["Year of Record"] - 2018) * -1
    return data


# Replace Gender with One Hot Encoding
def quantify_gender(data, train_data):
    allowed_values = ["male", "female", "other"]
    data.loc[~data["Gender"].isin(allowed_values), "Gender"] = "None"
    data["Gender"] = data["Gender"].replace({"None": None})
    data = pd.concat([data.drop("Gender", axis=1),
                      pd.get_dummies(data[["Gender"]])], axis=1)
    return data


# Replace unkonws in Age with mean
def quantify_age(data, train_data):
    data["Age"] = data["Age"].fillna(int((data["Age"].mean())))
    return data


# Replace Country with One Hot Encoding
def quantify_country(data, train_data):
    data["Country"] = data["Country"].fillna("None")
    allowed_values = train_data["Country"].unique()
    data.loc[~data["Country"].isin(allowed_values), "Country"] = "None"
    data["Country"] = data["Country"].replace({"None": None})
    data = pd.concat([data.drop("Country", axis=1),
                      pd.get_dummies(data[["Country"]])], axis=1)
    return data


# Replace unkowns in Size of City with mean
def quantify_size(data, train_data):
    data["Size of City"] = data["Size of City"].fillna(
        int((data["Size of City"].mean())))
    return data


# Replace Profession with One Hot Encoding
def quantify_profession(data, train_data):
    allowed_values = train_data["Profession"].unique()
    data.loc[~data["Profession"].isin(allowed_values), "Profession"] = "None"
    data["Profession"] = data["Profession"].replace({"None": None})
    data = pd.concat([data.drop("Profession", axis=1),
                      pd.get_dummies(data[["Profession"]])], axis=1)
    return data


# Rank PhD, Master, Bachelor and no degree respectively
def quantify_degree(data, train_data):
    data["University Degree"] = data["University Degree"].replace(
        {"PhD": 3,
         "Master": 2,
         "Bachelor": 1})
    allowed_values = ["PhD", "Master", "Bachelor"]
    data.loc[~data["University Degree"].isin(
        allowed_values), "University Degree"] = "None"
    data["University Degree"] = data["University Degree"].replace({"None": 0})
    return data


# Replace unkowns in Wears Glasses with mean
def quantify_glasses(data, train_data):
    data["Wears Glasses"] = data["Wears Glasses"].fillna(
        (data["Wears Glasses"].mean()))
    return data


# Replace Hair Color with One Hot Encoding
def quantify_hair(data, train_data):
    data["Hair Color"] = data["Hair Color"].fillna("None")
    allowed_values = train_data["Hair Color"].unique()
    data.loc[~data["Hair Color"].isin(allowed_values), "Hair Color"] = "None"
    data["Hair Color"] = data["Hair Color"].replace({"None": None})
    data = pd.concat([data.drop("Hair Color", axis=1),
                      pd.get_dummies(data[["Hair Color"]])], axis=1)
    return data


# Replace unkowns in Body Height with mean
def quantify_height(data, train_data):
    data["Body Height [cm]"] = data["Body Height [cm]"].fillna(
        (int(data["Body Height [cm]"].mean())))
    return data


def quantify_data(data, train_data):
    data = quantify_year(data, train_data)
    data = quantify_gender(data, train_data)
    data = quantify_age(data, train_data)
    data = quantify_country(data, train_data)
    data = quantify_size(data, train_data)
    data = quantify_profession(data, train_data)
    data = quantify_degree(data, train_data)
    data = quantify_glasses(data, train_data)
    data = quantify_hair(data, train_data)
    data = quantify_height(data, train_data)
    return data


# Remove outliers using standard deviation
def remove_outliers(data):
    data = data.dropna(subset=["Income"])
    data = data[np.abs(stats.zscore(data["Income"]) < 3)]

    data = data.dropna(subset=["Year of Record"])
    data = data[np.abs(stats.zscore(data["Year of Record"]) < 3)]

    data = data.dropna(subset=["Gender"])
    data = data = data[data["Gender"].isin(["male", "female", "other"])]

    data = data.dropna(subset=["Age"])
    data = data[np.abs(stats.zscore(data["Age"]) < 3)]

    data = data.dropna(subset=["Country"])

    data = data.dropna(subset=["Size of City"])
    data = data[np.abs(stats.zscore(data["Size of City"]) < 3)]

    data = data.dropna(subset=["Profession"])

    data = data.dropna(subset=["University Degree"])

    data = data.dropna(subset=["Wears Glasses"])
    data = data[data["Wears Glasses"].isin([1, 0])]

    data = data.dropna(subset=["Hair Color"])
    data = data = data[data["Hair Color"].isin(
        ["Black", "Brown", "Blond", "Red"])]

    data = data.dropna(subset=["Body Height [cm]"])
    data = data[np.abs(stats.zscore(data["Body Height [cm]"]) < 3)]
    return data
