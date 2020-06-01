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

train.drop(columns=['Id'], axis=1, inplace=True)
test.drop(columns=['Id'], axis=1, inplace=True)

# Saving the target values in "y_train".
y = train['SalePrice'].reset_index(drop=True)

previous_train = train.copy()

df = pd.concat((train, test)).reset_index(drop = True)

# Dropping the target variable.
df.drop(['SalePrice'], axis = 1, inplace = True)

print('Train shape: ', train.shape)
print('Test shape: ', test.shape)
print('All data shape: ', df.shape)

# #EDA
# ''' 1. Correlation
#     2. Distribution
#     3. Handle missing values
#     4. Feature Engineering
#     5. Modelling
#     6. Validation
#     7. Hyper parameter tuning
#     '''

# print('Initial type: ', train.MSSubClass.dtypes)
# print('Number of unique values: ', len(train['MSSubClass'].unique()))
# print('min: ', min(train.MSSubClass), 'max: ', max(train.MSSubClass))
# print('Number of Null values: ', train.MSSubClass.isnull().sum())


def type_tool(x, data):
    ''' function to describe a specific column in the data set'''
    n = data[x].dtypes
    m = len(data[x].unique())
    o = min(data[x])
    t = max(data[x])
    p = data[x].isnull().sum()
    d = {'col': [x], 'type': n, 'Uniques': m, 'Min': o, 'Max': t, 'Nulls': p}
    df = pd.DataFrame(data=d, index=[0])
    return df


def con_type(col, type, data):
    ''' Function to converst feature type '''
    data[col] = data[col].astype(type)
    return data[col]


def null_per(data):
    '''Calculates the percentage of missing features in a dataframe'''
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
    dtype = data.dtypes
    missing_data = pd.concat([total, percent, dtype], axis=1, keys=['Total', 'Percent', 'type'])
    return missing_data

def plot_missing(df):
    ''' Bar plot to show percentage of missing values in df dataframe and color indicates dtypes of features '''
    sns.set_style("dark")
    f, ax = plt.subplots(figsize=(8, 7))
    sns.set_color_codes(palette='deep')
    missing = null_per(df)
    missing = missing[missing.Percent > 0]
    missing.sort_values(by='Percent', inplace=True)
    g = sns.barplot(x=missing.index, y='Percent', hue='type', data=missing)
    g.set_xticklabels(g.get_xticklabels(), rotation=90)

print('Number of missing values: ', df.isna().sum().sum())
# plot_missing(df)
# plt.show()

convert_str = ['MSSubClass', 'YrSold', 'MoSold']
for i in convert_str:
    con_type(i, 'str', df)

df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))
df['MSZoning'] = df.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

mode_fill = ['Functional', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'SaleType', 'Electrical']
none_fill = ['Alley', 'MiscFeature', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']
zero_fill = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageArea', 'GarageCars', 'GarageYrBlt', 'MasVnrArea']
drop_col = ['Utilities', 'Street', 'PoolQC']

df = df.drop(drop_col, axis=1)

for i in mode_fill:
    df[i] = df[i].fillna(df[i].mode()[0])

for i in none_fill:
    df[i] = df[i].fillna('None')

for i in zero_fill:
    df[i] = df[i].fillna(0)

print('Number of missing values: ', df.isna().sum().sum())
print('All data after imputation: ', df.shape)

# Feature engineering more columns that can be grouped together

df['total_bsmt_sf'] = df['TotalBsmtSF']+df['1stFlrSF'] + df['2ndFlrSF']
df['yrblt_remod'] = df['YearBuilt'] + df['YearRemodAdd']
df['total_house_sf'] = df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['1stFlrSF'] + df['2ndFlrSF']
df['nb_baths'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])
df['porch'] = df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF']

print(df.shape)

df['haspool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
df['has2ndfloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
df['hasgarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
df['hasbsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
df['hasfireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

print(df.shape)

# Create a list of columns that need to be categories
print(df.dtypes.value_counts())