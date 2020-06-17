import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, median_absolute_error


class Linear_Regression():
    def __init__(self, df, correlation_value):
        self.predictors = []
        self.df = df
        self.correlation_value = correlation_value

    def correlation_assessment(self):
        self.df.corr()[['meantempm']].sort_values('meantempm')

    def print_df_corr(self):
        print(self.df.corr())

    def get_suitable_predictors(self):
        corr_values = self.df.corr()[['meantempm']]
        corr_values_processed = corr_values.unstack()
        var_names = []
        count = 0

        for key, value in corr_values_processed.items():
            var_names.append(key[1])
            print(key[1])
        for key, value in corr_values_processed.items():
            if value >= 0.6:
                self.predictors.append(var_names[count])
            count = count + 1
        if 'maxtempm' in self.predictors:
            self.predictors.remove('maxtempm')
        if 'mintempm' in self.predictors:
            self.predictors.remove('mintempm')
        if 'meantempm' in self.predictors:
            self.predictors.remove('meantempm')
        df2 = self.df[['meantempm'] + self.predictors]

        if len(self.predictors) > 3:
            return df2
        else:
            return False

    def plot_graph(self, df2):
        plt.rcParams['figure.figsize'] = [11, 15]

        fig, axes = plt.subplots(nrows=4, ncols=3, sharey=True)

        arr = np.array(self.predictors).reshape(4, 3)

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

    def further_filtering_the_predictors(self, alpha, df2):
        x = df2[self.predictors]
        y = df2['meantempm']

        x = sm.add_constant(x)
        # print("This is x:")
        # print(x)

        while True:
            model = sm.OLS(y, x).fit()
            model_pvalues_list = model.pvalues.tolist()
            max_value = max(model_pvalues_list)

            if alpha >= max_value:
                index_of_max_value = model_pvalues_list.index(max_value)
                variable_removed = self.predictors[index_of_max_value]
                x = x.drop(variable_removed, axis=1)
            else:
                return x

    def print_results(self, x, y):
        x = x.drop('const', axis=1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)

        regressor = LinearRegression()

        # fit the build the model by fitting the regressor to the training data
        regressor.fit(x_train, y_train)

        # make a prediction set using the test set
        prediction = regressor.predict(x_test)

        # Evaluate the prediction accuracy of the model
        print("The Explained Variance: %.2f" % regressor.score(x_test, y_test))
        print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_test, prediction))
        print("The Median Absolute Error: %.2f degrees celsius" % median_absolute_error(y_test, prediction))
