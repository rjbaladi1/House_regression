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


# print(null_per(train))
# print('Number of missing values: ', train.isna().sum().sum())

# Columns to drop

train.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Alley'], axis=1, inplace=True)
# print(null_per(train).sort_values(by=['Total','type'], ascending=[False, True]).head(50))
# print('Number of missing values: ', train.isna().sum().sum())




# Examining outliers in numerical features
# The plot shows that we have outliers for the numerical features

numerical_features = ['LotFrontage', 'MasVnrArea' ]

plt.figure(figsize=(15,5))

features_to_examine = ['LotFrontage','MasVnrArea','GarageYrBlt']
temp = train[numerical_features]
colors=['','red','blue','green']
i=1
for col in temp.columns:
    plt.subplot(1,3,i)
    a1 = sns.boxplot(data=temp,y=col,color=colors[i])
    i+=1

print(train.LotFrontage.mean(), train.LotFrontage.median())
print(train.MasVnrArea.mean(), train.MasVnrArea.median())
