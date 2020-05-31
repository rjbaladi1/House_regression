import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Importing the data
# From the initial shapes we can deduce that the split was roughly 50-50
# Write a function to import the data

test = pd.read_csv(os.path.join('dataset','test.csv'))

train = pd.read_csv(os.path.join('dataset','train.csv'))
sub = pd.read_csv(os.path.join('submission','sample_submission.csv'))

# print(test.shape)
# print(sub.head())
#
# #EDA
# ''' 1. Correlation
#     2. Distribution
#     3. Handle missing values
#     4. Feature Engineering
#     5. Modelling
#     6. Validation
#     7. Hyper parameter tuning
#     '''

# print(train.head())
# print(train.shape)
# print(train.info())
# print(train.skew())


# print('Initial type: ', train.MSSubClass.dtypes)
# print('Number of unique values: ', len(train['MSSubClass'].unique()))
# print('min: ', min(train.MSSubClass), 'max: ', max(train.MSSubClass))
# print('Number of Null values: ', train.MSSubClass.isnull().sum())


def type_tool(x, data):
    n = data[x].dtypes
    m = len(data[x].unique())
    o = min(data[x])
    t = max(data[x])
    p = data[x].isnull().sum()
    d = {'col': [x], 'type': n, 'Uniques': m, 'Min': o, 'Max': t, 'Nulls': p}
    df = pd.DataFrame(data=d, index=[0])
    return df


# print(train.dtypes.value_counts())

date_cols = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MoSold', 'YrSold']
int_object = ['MSSubClass', 'OverallQual', 'OverallCond', 'GarageCars']


def con_type(col, type, data):
    data[col] = data[col].astype(type)
    return data[col]


con_type(date_cols, 'datetime64[ns]', train)
con_type(int_object, 'object', train)

# print(train.dtypes.value_counts())

# Handling the missing data

def null_per(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
    dtype = data.dtypes
    missing_data = pd.concat([total, percent, dtype], axis=1, keys=['Total', 'Percent', 'type'])
    return missing_data


print(null_per(train))
print('Number of missing values: ', train.isna().sum().sum())


# Visualize missing values

sns.set_style("dark")
f, ax = plt.subplots(figsize= (8,7))
sns.set_color_codes(palette='deep')
missing = null_per(train)
missing = missing[missing.Percent>0]
missing.sort_values(by='Percent', inplace=True)
g = sns.barplot(x=missing.index, y='Percent', hue='type', data=missing)
g.set_xticklabels(g.get_xticklabels(), rotation=90)



# Columns to drop with above 80 percent of missing values

train.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1, inplace=True)
# print(null_per(train).sort_values(by=['Total','type'], ascending=[False, True]).head(50))
print('Number of missing values: ', train.isna().sum().sum())


# Examining outliers in numerical features
# The plot shows that we have outliers for the numerical features

numerical_features = ['LotFrontage', 'MasVnrArea']

# print(train.LotFrontage.mean(), train.LotFrontage.median())
# print(train.MasVnrArea.mean(), train.MasVnrArea.median())

for col in numerical_features:
    train[col].fillna(train[col].median(), inplace=True)

print('Number of missing values: ', train.isna().sum().sum())

sns.set_style("dark")
f, ax = plt.subplots(figsize= (8,7))
sns.set_color_codes(palette='deep')
missing = null_per(train)
missing = missing[missing.Percent>0]
missing.sort_values(by='Percent', inplace=True)
g = sns.barplot(x=missing.index, y='Percent', hue='type', data=missing)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.show()

# Taking a look at the date columns



# Once the categories are grouped we can start analyzing and imputing the missing values
