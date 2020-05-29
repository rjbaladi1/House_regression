import pandas as pd
import os
#import seaborn as sns
#import matplotlib.pyplot as plt

# Importing the data
# From the initial shapes we can deduce that the split was roughly 50-50
# Write a function to import the data

test = pd.read_csv(os.path.join('dataset','test.csv'))

train = pd.read_csv(os.path.join('dataset','train.csv'))
sub = pd.read_csv(os.path.join('submission','sample_submission.csv'))



print(test.shape)
print(sub.head())

#EDA

print(train.dtypes.value_counts())
print(train.describe())
# _ = sns.distplot(train.SalePrice)
# plt.show()
# plt.clf()

#missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))