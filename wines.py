"""
Question:

'Chemically speaking, what types of wine are there? What predicts wine quality?'

The latter question appears to be a fairly standard multivariate regression problem -
The former is more geared towards a neural network approach although testing for collinearity
will also yield results.
"""

"""
REQUIREMENTS:

Developed with python 3.7.0, pandas 0.23.4, statsmodels 0.9.0, matplotlib 3.0.2,
numpy 1.15.0, seaborn 0.9.0, sklearn 0.20.0
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import tree
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.graphics.api as smg
from statsmodels.stats.weightstats import ttest_ind

####################################################################################################
# SOURCES

data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
schema_url = data_url + "winequality.names"
red_filename = "winequality-red.csv"
white_filename = "winequality-white.csv"

#initial import.  User may want to run their own analysis, so these have not been enclosed.
red_wines = pd.read_csv(data_url + red_filename, delimiter=";")
red_wines.name = "red"
white_wines = pd.read_csv(data_url + white_filename, delimiter=";")
white_wines.name = "white"
all_wines = pd.concat([red_wines, white_wines])

#yes, we could use quartiles.  But isolating the property that's actually desired seems more useful.
quality_reds = red_wines[red_wines["quality"]>=6]
quality_whites = white_wines[white_wines["quality"]>=6]
poor_reds = red_wines[red_wines["quality"]<=4]
poor_whites = white_wines[white_wines["quality"]<=4]

#default confidence level
ALPHA = 0.05
#column names - all equivalent
variables = red_wines.columns
#useful keymaps
UP_ARROW = "↑"
DOWN_ARROW = "↓"

####################################################################################################
# Plotting methods

def plot_wines(wines_a, wines_b):
    """
    plots two wine datasets on a log scale
    """
    sns.violinplot(data=wines_a, scale="width", width=1.3, color="crimson")
    sns.violinplot(data=wines_b, scale="width", width=1.3, color="gold")
    plt.xlabel("Attributes")
    plt.ylabel("Arbitrary units")
    plt.yscale("log")
    plt.legend()
    plt.show()

def plot_single_regression(winelist, var):
    """
    visualisation of single quadratic regression ref'd
    """
    minvar, maxvar = winelist[var].min(), winelist[var].max()
    fm = f"quality ~ Q('{var}') + I(Q('{var}') ** 2)"
    model = smf.ols(formula=fm, data=winelist).fit()
    model_x = np.linspace(minvar, maxvar, 100) #VAR
    model_y = [model.params[0] + model.params[1]* x + model.params[2] * x**2 for x in model_x]
    plt.plot(model_x, model_y)
    #trying some different methods here to represent underlying data density
    #plt.scatter(winelist[var], winelist["quality"], s=5000, color="grey", linewidth=0, alpha=0.01)
    sns.kdeplot(winelist[var], winelist["quality"], shade=True, cmap="Greys")
    plt.xlim(right=maxvar)
    plt.xlabel(var)
    plt.ylabel("quality") 
    plt.title(f"quadratic fit for {var} in {winelist.name} wines")   
    plt.show()

def tabulate_recipe(input_filename="recipe.csv"):
    def colorise(val):
        cmap = {UP_ARROW : "green", DOWN_ARROW : "red", "nan" : "grey"}
        color = cmap.get(val, "")
        return 'background-color: %s' % color
    recipes = pd.read_csv(input_filename)
    recipes = recipes.replace("nan", "")
    recipes = recipes.style.applymap(colorise).set_precision(2).set_properties(**{'text-align' : 'center'})
    with open("recipe.htm", "w") as f:
        html = recipes.render()
        f.write(html)

####################################################################################################

def compare_wines(wines_a, wines_b, filename = "comparison.csv"):
    #wines_a_data, wines_b_data = wines_a["data"], wines_b["data"]
    index=["var", "t", "P", "DoF"]
    t_test_results = pd.DataFrame(columns=index)
    for var in variables:
        s = pd.Series([var, *ttest_ind(wines_a[var], wines_b[var])], index=index)
        t_test_results = t_test_results.append(s, ignore_index=True)
    print(t_test_results)
    t_test_results.to_csv(filename)

def find_predictor(winelist, results_filename):
    """
    Runs OLS regression for each attribute individually (i.e. not testing for collinearity yet).
    """
    models, recipe = [], []
    print(f"finding predictors for: {winelist.name}")
    with open(results_filename, "w") as f:
        for var in variables[:-2]: #last column is red/white, second last the DV
            fm = f"quality ~ Q('{var}') + I(Q('{var}') ** 2)" 
            # minimum quality is 3, max is 9, so don't worry about setting intercepts
            model = smf.ols(formula=fm, data=winelist).fit()
            model.name = var
            # print(f"{var}\t{model.rsquared}")
            # (returns statsmodels.regression.linear_model.RegressionResultsWrapper)
            f.write(f"\n\ntesting : {var}\n")
            f.write(model.summary().as_text())
            action = get_action(var, model)
            models.append(model)
    return models, recipe

def get_action(var, model):
    def optimise(var, model):
        print(f"optimising... {var}")
        return np.roots(model.params)[1]
    if ((model.params[1] > 0 and model.pvalues[1] < ALPHA) and
        (model.params[2] < 0 and model.pvalues[2] < ALPHA)):
            #optimise
            roots = optimise(var, model)
            return roots
    elif (model.pvalues[1] < ALPHA):
        return (UP_ARROW if model.params[1] > 0 else DOWN_ARROW)
    else:
        return None

###################################################################################################

def test_collinear(winelist):
    corr_matrix = np.corrcoef(winelist["data"])
    smg.plot_corr(corr_matrix)
    plt.show()

if __name__ == "__main__":
    #plot_wines(quality_reds, poor_reds)
    compare_wines(quality_reds, poor_reds)
    red_predictor, white_predictor = (find_predictor(red_wines, "red_model_results.txt"),
                                      find_predictor(white_wines, "white_model_results.txt"))
    #for var in variables:
    #    plot_single_regression(red_wines, var)
    #    plot_single_regression(white_wines, var)
    #tabulate_recipe()




