# Replace NaN in Year of Record with mean and convert to age of record
def quantify_year(data, train_data):
    data["Year of Record"] = data["Year of Record"].fillna(
        (data["Year of Record"].mean()))
    data["Year of Record"] = (data["Year of Record"] - 2019) * -1
    return data


# Replace Gender with mean Income for male, female and other
def quantify_gender(data, train_data):
    for gender in ["male", "female", "other"]:
        income = train_data.loc[train_data["Gender"]
                                == gender]["Income"].mean()
        data["Gender"] = data["Gender"].replace({gender: income})
    for gender in ["0", "unknown"]:
        income = train_data["Income"].mean()
        data["Gender"] = data["Gender"].replace({gender: income})
    data["Gender"] = data["Gender"].fillna(train_data["Income"].mean())
    return data


# Replace NaN in Age with mean
def quantify_age(data, train_data):
    data["Age"] = data["Age"].fillna((data["Age"].mean()))
    return data


# Replace Country with mean Income for that country
def quantify_country(data, train_data):
    for country in data["Country"].unique():
        income = train_data.loc[train_data["Country"]
                                == country]["Income"].mean()
        data["Country"] = data["Country"].replace({country: income})
    data["Country"] = data["Country"].fillna(train_data["Income"].mean())
    return data


# Replace NaN in Size of City with mean
def quantify_size(data, train_data):
    data["Size of City"] = data["Size of City"].fillna(
        (data["Size of City"].mean()))
    return data


# Replace Profession with mean Income for that profession
def quantify_profession(data, train_data):
    for profession in data["Profession"].unique():
        income = train_data.loc[train_data["Profession"]
                                == profession]["Income"].mean()
        data["Profession"] = data["Profession"].replace({profession: income})
    data["Profession"] = data["Profession"].fillna(train_data["Income"].mean())
    return data


# Rank PhD, Master, Bachelor and no degree respectively
def quantify_degree(data, train_data):
    data["University Degree"] = data["University Degree"].replace(
        {"PhD": 3,
         "Master": 2,
         "Bachelor": 1})
    for degree in ["No", "0"]:
        data["University Degree"] = data["University Degree"].replace(
            {degree: 0})
    data["University Degree"] = data["University Degree"].fillna(0)
    return data


# Replace NaN in Wears Glasses with mean
def quantify_glasses(data, train_data):
    data["Wears Glasses"] = data["Wears Glasses"].fillna(
        (data["Wears Glasses"].mean()))
    return data


# Replace Hair Color with mean Income for that hair color
def quantify_hair(data, train_data):
    for color in data["Hair Color"].unique():
        income = train_data.loc[train_data["Hair Color"]
                                == color]["Income"].mean()
        data["Hair Color"] = data["Hair Color"].replace({color: income})
    data["Hair Color"] = data["Hair Color"].fillna(train_data["Income"].mean())
    return data


# Replace NaN in Body Height with mean
def quantify_height(data, train_data):
    data["Body Height [cm]"] = data["Body Height [cm]"].fillna(
        (data["Body Height [cm]"].mean()))
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
