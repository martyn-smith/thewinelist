"""
Question:

'Chemically speaking, what types of wine are there? What predicts wine quality?'

The latter question appears to be a fairly standard multivariate regression problem -
The former is more geared towards a neural network approach although testing for collinearity
will also yield results.

TODO: 
-Bayesian methods ("find_predictors" is really more like optimisiation,
good for industry but doesn't answer "good/bad if X" well).
-Font wrapping.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
#from sklearn import tree
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.graphics.api as smg
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

####################################################################################################
# SOURCES

data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
schema_url = data_url + "winequality.names"
red_filename = "winequality-red.csv"
white_filename = "winequality-white.csv"

####################################################################################################
# initial import.  User may want to run their own analysis, so these have not been enclosed.
red_wines = pd.read_csv(data_url + red_filename, delimiter=";")
red_wines.name = "red"
white_wines = pd.read_csv(data_url + white_filename, delimiter=";")
white_wines.name = "white"

#yes, we could use quartiles.  But isolating the property that's actually desired seems more useful.
quality_reds = red_wines[red_wines["quality"]>=6]
quality_reds.name = "quality_reds"
quality_whites = white_wines[white_wines["quality"]>=6]
quality_whites.name = "quality_whites"
poor_reds = red_wines[red_wines["quality"]<=4]
poor_reds.name = "poor_reds"
poor_whites = white_wines[white_wines["quality"]<=4]
poor_whites.name = "poor_whites"

#default confidence level, set a little low.
ALPHA = 0.01
#column names - all equivalent.  Could use any of the dataframes if desired.
variables = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
             'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
             'pH', 'sulphates', 'alcohol', 'quality']
#useful keymaps
UP_ARROW = "↑"
DOWN_ARROW = "↓"

####################################################################################################
# Plotting methods

def plot_wines():
    red_melted = pd.melt(red_wines)
    white_melted = pd.melt(white_wines)
    df = pd.concat([red_melted.assign(variant="red"), white_melted.assign(variant="white")])
    sns.violinplot(x="variable", y="value", data=df, scale="width", hue="variant", split=True,
                   palette=["crimson", "gold"])
    plt.xlabel("Attributes")
    plt.ylabel("Arbitrary units")
    plt.yscale("log")
    plt.legend()
    plt.show()

def plot_single_regression(winelist, var):
    """
    visualisation of single quadratic regression.
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

def plot_corr(red_corr, white_corr):
    smg.plot_corr(red_corr, xnames=variables, ynames=variables, 
                  cmap="Reds", normcolor=True) 
    smg.plot_corr(white_corr, xnames=variables, ynames=variables, 
                  cmap="YlGn", normcolor=True)
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
# CALCULATIONS

def compare_wines(wines_a, wines_b, results_filename = "_comparison.csv"):
    print(f"comparing {wines_a.name} and {wines_b.name}")
    index=["var", "t", "P", "DoF"]
    t_test_results = pd.DataFrame(columns=index)
    for var in variables:
        s = pd.Series([var, *ttest_ind(wines_a[var], wines_b[var])], index=index)
        t_test_results = t_test_results.append(s, ignore_index=True)
    print(t_test_results)
    t_test_results.to_csv(wines_a.name + wines_b.name + results_filename)

def test_collinear(winelist):
    print(f"testing collinearity for {winelist.name}")
    for i, var in enumerate(variables): 
        print(f"{var} VIF is: {vif(winelist.values, i)}")
    corr_matrix = np.corrcoef(winelist, rowvar=False)
    return corr_matrix

def compare_quality_to_poor():
    """
    Simple independent t-test on quality and poor wines (closest thing to true predictors here).
    """
    print("comparing quality and poor red wines")
    compare_wines(quality_reds, poor_reds)
    print(quality_reds.mean() - poor_reds.mean())
    print("comparing quality and poor white wines")
    compare_wines(quality_whites, poor_whites)
    print(quality_whites.mean() - quality_whites.mean())

def find_predictor(winelist, results_filename="_model_results.txt"):
    """
    Runs OLS regression for each attribute individually (i.e. not testing for collinearity yet),
    based on a quadratic fit.
    """
    print(f"finding predictors for: {winelist.name}")
    models, recipe = [], []
    with open(winelist.name + results_filename, "w") as f:
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
    """
    Simple utility to find an optimum (if possible), or state whether var should be 
    maximised or minimised.
    """
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

if __name__ == "__main__":
    plot_wines()
    compare_quality_to_poor()
    red_corr, white_corr = test_collinear(red_wines), test_collinear(white_wines)
    plot_corr(red_corr, white_corr)
    red_predictor, white_predictor = (find_predictor(red_wines),
                                      find_predictor(white_wines))
    #for var in variables:
    #    plot_single_regression(red_wines, var)
    #    plot_single_regression(white_wines, var)
    #tabulate_recipe()
    





