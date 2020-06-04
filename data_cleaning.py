from collections import namedtuple
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import statsmodels.api as sm

variables = ["date", "meantempm", "meandewptm", "meanpressurem", "maxhumidity", "minhumidity", "maxtempm",
            "mintempm", "maxdewptm", "mindewptm", "maxpressurem", "minpressurem", "precipm"]

records = []

AnnualWeatherReport = namedtuple("AnnualWeatherReport", variables)

with open("raw_data.txt", "r") as raw_data:
    r = csv.reader(raw_data, delimiter=",")
    next(r)
    for row in r:
        records.append(AnnualWeatherReport(
            date=row[0],
            meantempm=row[1],
            meandewptm=row[2],
            meanpressurem=row[3],
            maxhumidity=row[4],
            minhumidity=row[5],
            maxtempm=row[6],
            mintempm=row[7],
            maxdewptm=row[8],
            mindewptm=row[9],
            maxpressurem=row[10],
            minpressurem=row[11],
            precipm=row[12]
        ))

df = pd.DataFrame(records, columns=variables).set_index('date')

# Alternative
# days_prior = 4

# def add_nth_day_prior_features(df, days_prior):
#     for target_variable in variables:
#         if target_variable != 'date':
#             for i in range(1, 4):
#                 rows = df.shape[0]
#                 nth_prior_values = [None]*days_prior + [df[target_variable][i-days_prior] for i in range
#                 (days_prior, rows)]
#                 col_name = "{}_{}".format(target_variable, days_prior)
#                 df[col_name] = nth_prior_values
#
#
# add_nth_day_prior_features(df, days_prior)


def add_nth_day_prior_features(df, target_variable, days_prior):
    rows = df.shape[0]
    nth_prior_values = [None]*days_prior + [df[target_variable][i-days_prior] for i in range(days_prior, rows)]
    col_name = "{}_{}".format(target_variable, days_prior)
    df[col_name] = nth_prior_values


for target_variable in variables:
    if target_variable != 'date':
        for i in range(1, 4):
            add_nth_day_prior_features(df, target_variable, i)


variables_removed = [variable for variable in variables if variable not in ['meantempm', 'mintempm', 'maxtempm']]
variables_maintained = [col for col in df.columns if col not in variables_removed]

df = df[variables_maintained]

df = df.apply(pd.to_numeric, errors='coerce')
print(df.info())

spread = df.describe().T
IQR = spread['75%'] - spread['25%']
spread['outliers'] = (spread['min']<(spread['25%']-(3*IQR)))|(spread['max'] > (spread['75%']+3*IQR))
print(spread.loc[spread.outliers])

plt.rcParams['figure.figsize'] = [14, 8]
df.maxhumidity_1.hist()
plt.title('Distribution of maxhumidity_1')
plt.xlabel('maxhumidity_1')
plt.show()

df.minpressurem_1.hist()
plt.title('Distribution of minpressurem_1')
plt.xlabel('minpressurem_1')
plt.show()

# If Needed
for precip_col in ['precipm_1', 'precipm_2', 'precipm_3']:
    # create a boolean array of values representing nans
    missing_vals = pd.isnull(df[precip_col])
    df[precip_col][missing_vals] = 0

df = df.dropna()

df.corr()[['meantempm']].sort_values('meantempm')
print(df.corr())
predictors = ['meantempm_1',  'meantempm_2',  'meantempm_3',
              'mintempm_1',   'mintempm_2',   'mintempm_3',
              'meandewptm_1', 'meandewptm_2', 'meandewptm_3',
              'maxdewptm_1',  'maxdewptm_2',  'maxdewptm_3',
              'mindewptm_1',  'mindewptm_2',  'mindewptm_3',
              'maxtempm_1',   'maxtempm_2',   'maxtempm_3']
df2 = df[['meantempm'] + predictors]

#%matplotlib inline

# manually set the parameters of the figure to and appropriate size
plt.rcParams['figure.figsize'] = [16, 22]

# call subplots specifying the grid structure we desire and that
# the y axes should be shared
fig, axes = plt.subplots(nrows=6, ncols=3, sharey=True)

# Since it would be nice to loop through the features in to build this plot
# let us rearrange our data into a 2D array of 6 rows and 3 columns
arr = np.array(predictors).reshape(6, 3)

# use enumerate to loop over the arr 2D array of rows and columns
# and create scatter plots of each meantempm vs each feature
for row, col_arr in enumerate(arr):
    for col, feature in enumerate(col_arr):
        axes[row, col].scatter(df2[feature], df2['meantempm'])
        if col == 0:
            axes[row, col].set(xlabel=feature, ylabel='meantempm')
        else:
            axes[row, col].set(xlabel=feature)
plt.show()

# import the relevant module

