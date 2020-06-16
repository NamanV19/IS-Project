from collections import namedtuple
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


class LinearRegression():
    def __init__(self, df, correlation_value, alpha):
        self.predictors = []
        self.df = df
        self.correlation_value = correlation_value
        self.alpha = alpha

    def correlation_assessment(self):
        self.df.corr()[['meantempm']].sort_values('meantempm')

    def print_df_corr(self):
        print(self.df.corr())

    def get_suitable_predictors(self):
        corr_values = self.df.corr().sort_values('meantempm').abs()
        corr_values_processed = corr_values.unstack()
        for i in range(len(corr_values_processed)):
            if corr_values_processed[i] > self.correlation_value:
                self.predictors.append(corr_values_processed[1])
        return self.predictors
        # for key, value in corr_values_processed.items():
        #     if value>=self.correlation_value:
        #         self.predictors.append(key[1])
        return self.predictors
        if 'maxtempm' in self.predictors:
            self.predictors.remove('maxtempm')
        if 'mintempm' in self.predictors:
            self.predictors.remove('mintempm')
        df2 = self.df[['meantempm'] + self.predictors]

        if len(self.predictors) > 3:
            return df2
        else:
            return False

    def plot_graph(self):
        df2 = self.get_suitable_predictors();

        plt.rcParams['figure.figsize'] = [16, 22]

        fig, axes = plt.subplots(nrows=6, ncols=3, sharey=True)

        arr = np.array(self.predictors).reshape(6, 3)

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

    def further_filtering_the_predictors(self, alpha):
        df2 = self.get_suitable_predictors()
        x = df2[self.predictors]
        y = df2['meantempm']

        x = sm.add_constant(x)

        model = sm.OLS(y, x).fit()
        max_value = max(model.pvalues)

        while True:
            if alpha >= max_value:
                x = x.drop('', axis=1)
                model = sm.OLS(y, x).fit()
                max_value = max(model.pvalues)
            else:
                return x

    # def print_results(self, x):
    #     x = x.drop('const', axis=1)
    #     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)
    #
    #     regressor = LinearRegression()
    #
    #     # fit the build the model by fitting the regressor to the training data
    #     regressor.fit(x_train, y_train)
    #
    #     # make a prediction set using the test set
    #     prediction = regressor.predict(x_test)
    #
    #     # Evaluate the prediction accuracy of the model
    #     from sklearn.metrics import mean_absolute_error, median_absolute_error
    #     print("The Explained Variance: %.2f" % regressor.score(x_test, y_test))
    #     print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_test, prediction))
    #     print("The Median Absolute Error: %.2f degrees celsius" % median_absolute_error(y_test, prediction))


def main():
    variables = ["date", "meantempm", "meandewptm", "meanpressurem", "maxhumidity", "minhumidity", "maxtempm",
                 "mintempm", "maxdewptm", "mindewptm", "maxpressurem", "minpressurem", "precipm"]

    records = []

    AnnualWeatherReport = namedtuple("AnnualWeatherReport", variables)

    with open("ottawa_raw_data.txt", "r") as raw_data:
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

    def add_nth_day_prior_features(df, target_variable, days_prior):
        rows = df.shape[0]
        nth_prior_values = [None] * days_prior + [df[target_variable][i - days_prior] for i in range(days_prior, rows)]
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
    # print(df.info())

    spread = df.describe().T
    IQR = spread['75%'] - spread['25%']
    spread['outliers'] = (spread['min'] < (spread['25%'] - (3 * IQR))) | (spread['max'] > (spread['75%'] + 3 * IQR))
    # print(spread.loc[spread.outliers])

    plt.rcParams['figure.figsize'] = [14, 8]
    df.maxhumidity_1.hist()
    plt.title('Distribution of maxhumidity_1')
    plt.xlabel('maxhumidity_1')
    # plt.show()

    df.minpressurem_1.hist()
    plt.title('Distribution of minpressurem_1')
    plt.xlabel('minpressurem_1')
    # plt.show()

    # If Needed
    for precip_col in ['precipm_1', 'precipm_2', 'precipm_3']:
        # create a boolean array of values representing nans
        missing_vals = pd.isnull(df[precip_col])
        df[precip_col][missing_vals] = 0

    df = df.dropna()

    ln = LinearRegression(df, 0.6, 0.05)

    ln.correlation_assessment()
    ln.print_df_corr()
    print(ln.get_suitable_predictors())

main()













