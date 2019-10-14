import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer


def remove_outliers(data):
    # Drop unknowns and outliers beyond 3 standard deviations from Income
    if "Income" in data.columns:
        data = data.dropna(subset=["Income"])
        data = data[np.abs(stats.zscore(data["Income"]) < 3)]

    # Drop unknowns and outliers beyond 3 standard deviations from Year
    if "Year of Record" in data.columns:
        data = data.dropna(subset=["Year of Record"])
        data = data[np.abs(stats.zscore(data["Year of Record"]) < 3)]

    # Drop unknowns from Gender
    if "Gender" in data.columns:
        data = data.dropna(subset=["Gender"])
        data = data = data[data["Gender"].isin(["male", "female", "other"])]

    # Drop unknowns and outliers beyond 3 standard deviations from Age
    if "Age" in data.columns:
        data = data.dropna(subset=["Age"])
        data = data[np.abs(stats.zscore(data["Age"]) < 3)]

    # Drop unknowns from Country
    if "Country" in data.columns:
        data = data.dropna(subset=["Country"])

    # Drop unknowns and outliers beyond 3 standard deviations from Size of City
    if "Size of City" in data.columns:
        data = data.dropna(subset=["Size of City"])
        data = data[np.abs(stats.zscore(data["Size of City"]) < 3)]

    # Drop unknowns from Profession
    if "Profession" in data.columns:
        data = data.dropna(subset=["Profession"])

    # Drop unknowns from University Degree
    if "University Degree" in data.columns:
        data = data.dropna(subset=["University Degree"])

    # Drop unknowns from Wears Glasses
    if "Wears Glasses" in data.columns:
        data = data.dropna(subset=["Wears Glasses"])
        data = data[data["Wears Glasses"].isin([1, 0])]

    # Drop unknowns from Hair Color
    if "Hair Color" in data.columns:
        data = data.dropna(subset=["Hair Color"])
        data = data = data[data["Hair Color"].isin(
            ["Black", "Brown", "Blond", "Red"])]

    # Drop unknowns and outliers beyond 3 standard deviations from Body Height
    if "Body Height [cm]" in data.columns:
        data = data.dropna(subset=["Body Height [cm]"])
        data = data[np.abs(stats.zscore(data["Body Height [cm]"]) < 3)]

    return data


def quantify_data(data):
    # Replace unknowns in Year of Record with mean and convert to age of record
    if "Year of Record" in data.columns:
        data["Year of Record"] = data["Year of Record"].fillna(
            int(data["Year of Record"].mean()))
        data["Year of Record"] = (data["Year of Record"] - 2018) * -1

    # Replace Gender with One Hot Encoding
    if "Gender" in data.columns:
        allowed_values = ["male", "female", "other"]
        data.loc[~data["Gender"].isin(allowed_values), "Gender"] = "None"
        data["Gender"] = data["Gender"].replace({"None": None})
        data = pd.concat([data.drop("Gender", axis=1),
                          pd.get_dummies(data[["Gender"]])], axis=1)

    # Replace unknowns in Age with mean
    if "Age" in data.columns:
        data["Age"] = data["Age"].fillna(int((data["Age"].mean())))

    # Replace Country with One Hot Encoding
    if "Country" in data.columns:
        data["Country"] = data["Country"].fillna("None")
        data["Country"] = data["Country"].replace({"None": None})
        data = pd.concat([data.drop("Country", axis=1),
                          pd.get_dummies(data[["Country"]])], axis=1)

    # Replace unknowns in Size of City with mean
    if "Size of City" in data.columns:
        data["Size of City"] = data["Size of City"].fillna(
            int((data["Size of City"].mean())))

    # Replace Profession with One Hot Encoding
    if "Profession" in data.columns:
        data["Profession"] = data["Profession"].replace({"None": None})
        data = pd.concat([data.drop("Profession", axis=1),
                          pd.get_dummies(data[["Profession"]])], axis=1)

    # Rank PhD, Master, Bachelor and no degree respectively
    if "University Degree" in data.columns:
        data["University Degree"] = data["University Degree"].replace(
            {"PhD": 3,
             "Master": 2,
             "Bachelor": 1})
        allowed_values = ["PhD", "Master", "Bachelor"]
        data.loc[~data["University Degree"].isin(
            allowed_values), "University Degree"] = "None"
        data["University Degree"] = data["University Degree"].replace(
            {"None": 0})

    # Replace unknowns in Wears Glasses with mean
    if "Wears Glasses" in data.columns:
        data["Wears Glasses"] = data["Wears Glasses"].fillna(
            data["Wears Glasses"].mean())

    # Replace Hair Color with One Hot Encoding
    if "Hair Color" in data.columns:
        data["Hair Color"] = data["Hair Color"].fillna("None")
        data["Hair Color"] = data["Hair Color"].replace({"None": None})
        data = pd.concat([data.drop("Hair Color", axis=1),
                          pd.get_dummies(data[["Hair Color"]])], axis=1)

    # Replace unknowns in Body Height with mean
    if "Body Height [cm]" in data.columns:
        data["Body Height [cm]"] = data["Body Height [cm]"].fillna(
            int(data["Body Height [cm]"].mean()))

    return data


def polynomialize_data(data):
    transformer = PolynomialFeatures(degree=2)
    input = data[["Year of Record", "Age", "Size of City", "Body Height [cm]"]]
    polynomial_data = transformer.fit_transform(input)
    polynomial_columns = transformer.get_feature_names(input.columns)
    polynomial_data = pd.DataFrame(polynomial_data, columns=polynomial_columns)
    polynomial_data = polynomial_data.drop(input.columns, axis=1)
    data = pd.concat(
        [data.reset_index(), polynomial_data.reset_index()], axis=1)
    return data


def standardize_data(data):
    transformer = StandardScaler()
    data = transformer.fit_transform(data)
    return data


def normalize_data(data):
    transformer = Normalizer().fit(data)
    data = transformer.fit_transform(data)
    return data
