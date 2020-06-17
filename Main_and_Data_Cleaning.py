from collections import namedtuple
from linear_regression import Linear_Regression
import matplotlib.pyplot as plt
import pandas as pd
import csv


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

    ln = Linear_Regression(df, 0.6)

    ln.correlation_assessment()
    ln.print_df_corr()
    df2 = ln.get_suitable_predictors()
    ln.plot_graph(df2)
    x = ln.further_filtering_the_predictors(0.05, df2)
    y = df2['meantempm']
    ln.print_results(x, y)


main()













