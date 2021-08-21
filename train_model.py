from src.data.make_dataset import partition_dataset
from src.features.build_features import pca_dim_reduction
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

###############################################################################
################# Initializing functions and variables ########################
###############################################################################

# Number of hyper parameter combinations to run in random search
# Values set to 1 to make the script quick to run for Jacob / Atif
random_seach_n = 1 # 500 used for report
random_search_n_svr = 1 # only 100 used for report due to fit time complexity

# Initialize these before you train and evaluate the models
def train_test_regressor(x_train, x_test, y_train, y_test, description, regressor):
    """
    Procedure for training and test a model. Was only used initially, until I 
    decided to use cross validation.

    Parameters
    ----------
    x_train : Pandas df or series 
        Predictor variables, training split
        
    x_test : Pandas df or series 
        Predictor variables, test split
        
    y_train : Pandas df or series 
        Target variable, training split
        
    y_test : Pandas df or series 
        Target variable, test split
        
    description : STR
        String describing the model. Used for print out of performance
    
    regressor : model
        Any model compatible with a fit() and a predict() method

    Returns
    -------
    regressor : TYPE
        DESCRIPTION.
    rmse : TYPE
        DESCRIPTION.

    """
    regressor.fit(x_train, y_train)
    y_predicted = regressor.predict(x_test)
    mse = mean_squared_error(y_test, y_predicted)
    rmse = mean_squared_error(y_test, y_predicted, squared=False)
    mae = mean_absolute_error(y_test, y_predicted)
    print("Model: {} \nRMSE: {} \nMSE: {} \nMAE: {}".format(description, rmse, mse, mae))
    return regressor, rmse

def cross_validate(df, regressor, target_variable):
    """
    Using k-fold CV to evaluate model performance. I use the whole dataset 
    because the regressor is not fit beforehand - only during each fold. 
    I.e. there is no data snooping.

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe containing all predictor variables and the target variable
        
    regressor : regression model
        Any model compatible with sklearn.cross_val_score
        
    target_variable : STR
        The name of the target variable

    Returns
    -------
    None.

    """
    print("Starting cross validation of: \n{}".format(regressor))
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    cv_x = df.drop(target_variable, axis=1)
    cv_y = df[target_variable]
    n_rmse = cross_val_score(regressor, 
                               cv_x, 
                               cv_y,
                               scoring="neg_root_mean_squared_error",
                               cv=cv, 
                               n_jobs=-1, 
                               error_score='raise')
    
    n_mae = cross_val_score(regressor, 
                            cv_x, 
                            cv_y,
                            scoring="neg_mean_absolute_error",
                            cv=cv, 
                            n_jobs=-1, 
                            error_score='raise')
    
    print("Mean negative RMSE from 10 fold CV: \n {}".format(np.mean(n_rmse)))
    print("Mean negative MAE from 10 fold CV: \n {}".format(np.mean(n_mae)))

# Define target variable - either "list_price_dkk" or "m2_price"
target_variable = "m2_price" 

if target_variable == "m2_price":
    # This dataset does not contain the list prices
    #df = pd.read_csv("C:/Users/groes/OneDrive/Documents/701CW_2/Intro_to_AI/data/processed/3.0-processed_data.csv")
    df = pd.read_csv("data/processed/3.0-processed_data.csv")
else:
    #df = pd.read_csv("C:/Users/groes/OneDrive/Documents/701CW_2/Intro_to_AI/data/processed/4.0-processed_data_w_listprice.csv")
    df = pd.read_csv("data/processed/4.0-processed_data_w_listprice.csv")
    df.drop("m2_price", axis=1, inplace=True) # avoiding data snooping

# Dropping index column created when saving the df
try:
    df.drop("Unnamed: 0" , axis=1, inplace=True)
except:
    print("Column 'Unnamed: 0' not found in df")

# Not used to train/evaluate model but to epxeriment with various hyperparameter values
# Dataset partitioning without any scaling 
x_train, x_test, y_train, y_test = partition_dataset(df, target_name=target_variable, 
                                                     training_size=0.8)

# I trained a few models without the one hot encoded zip code variables to see the effect
zipcode_columns = [col for col in df.columns if "Zip" in col]
df_no_zipcodes = df.drop(zipcode_columns, axis=1)



###############################################################################
################################# Linear Regression ###########################
###############################################################################


# Linear regression without PCA
linear_regressor = LinearRegression()
_ = train_test_regressor(
    x_train,
    x_test, 
    y_train, 
    y_test, 
    "linreg, no PCA", 
    linear_regressor)


# Linear regression with PCA
x_train_pca = pca_dim_reduction(x_train, explained_variance=0.95)
x_test_pca = pca_dim_reduction(x_test, explained_variance=0.95)
model = train_test_regressor(
    x_train_pca, 
    x_test_pca, 
    y_train, 
    y_test, 
    "linreg with PCA", 
    linear_regressor)

# Linreg with standardized data
numerical_variables = ["rooms", "home_size_m2", "lotsize_m2", "expenses_dkk", "zipcode_avg_m2_price", "age", "balcony", ]
df_numerical = df[numerical_variables]
scaler = StandardScaler()
scaler.fit(df_numerical)
standardized = scaler.transform(df_numerical)
df_standardized = pd.DataFrame(standardized, columns=numerical_variables)
df_unscaled = df.drop(numerical_variables, axis=1)
df_standardized = pd.concat([df_standardized, df_unscaled], axis=1)
x_train_standardized, x_test_standardized, y_train_standardized, y_test_standardized \
    = partition_dataset(df_standardized, target_name=target_variable, training_size=0.8)
    
model = train_test_regressor(
    x_train_standardized, 
    x_test_standardized,
    y_train_standardized,
    y_test_standardized,
    "linreg with standardized data",
    linear_regressor
    )
    
# Linreg with data standardization before PCA
threshold = 0.95 # I experimented with several values as described in the report
x_train_pca = pca_dim_reduction(x_train_standardized, explained_variance=threshold)
x_test_pca = pca_dim_reduction(x_test_standardized, explained_variance=threshold)
model = train_test_regressor(
    x_train_pca, 
    x_test_pca, 
    y_train, 
    y_test, 
    "linreg with standardized data and pca",
    linear_regressor
    )

# Linreg with min-max scaled data
scaler = MinMaxScaler()
scaler.fit(df_numerical)
min_max_scaled = scaler.transform(df_numerical)
df_min_max_scaled = pd.DataFrame(min_max_scaled, columns=numerical_variables)
df_min_max_scaled = pd.concat([df_min_max_scaled, df_unscaled], axis=1)
x_train_mm_scaled, x_test_mm_scaled, y_train_mm_scaled, y_test_mm_scaled \
    = partition_dataset(df_min_max_scaled, target_name=target_variable,
                        training_size=0.8)
    
model = train_test_regressor(
    x_train_mm_scaled,
    x_test_mm_scaled, 
    y_train_mm_scaled, 
    y_test_mm_scaled, 
    "linreg with min-max scaled data",
    linear_regressor
    )

# Linreg with min-max scaling before PCA
threshold = 0.8 # I experimented with several values for this as described in report
x_train_pca_minmax = pca_dim_reduction(x_train, explained_variance=threshold)
x_test_pca_minmax = pca_dim_reduction(x_test, explained_variance=threshold)
model = train_test_regressor(
    x_train_pca_minmax, 
    x_test_pca_minmax, 
    y_train, 
    y_test, 
    "linreg with standardized data and pca",
    linear_regressor
    )


cross_validate(df, linear_regressor, target_variable)

###############################################################################
#################################### Baseline #################################
###############################################################################
baseline_rmse = mean_squared_error(df["m2_price"], df["zipcode_avg_m2_price"], squared=False)
baseline_mse = mean_squared_error(df["m2_price"], df["zipcode_avg_m2_price"])
baseline_mae = mean_absolute_error(df["m2_price"], df["zipcode_avg_m2_price"])
print("Model: {} \n RMSE: {} \n MSE: {} \nMAE: {}".format(
    "Baseline", baseline_rmse, baseline_mse, baseline_mae))


###############################################################################
####################### Support vector regression #############################
###############################################################################

# Trying different kernel types
kernels = ["linear", "poly", "rbf", "sigmoid"]

for kernel in kernels:
    # Support vector regression - data not scaled
    description = "Support vector regression, unscaled data. Kernel=" + kernel
    svr = SVR(kernel) # default kernel='RBF'
    _ = train_test_regressor(x_train, x_test, y_train, y_test, description, svr)
    
    # Support vector regression - standardized data
    description = "Support vector regression, standardized data. Kernel=" + kernel
    svr = SVR(kernel) # default kernel='RBF'
    _ = train_test_regressor(
        x_train_standardized, 
        x_test_standardized, 
        y_train, 
        y_test, 
        description, 
        svr
        )
    
    # Support vector regression - min-max scaled data
    description = "Support vector regression, min-max scaled data. Kernel=" + kernel
    svr = SVR(kernel) # default kernel='RBF'
    _ = train_test_regressor(
        x_train_mm_scaled, 
        x_test_mm_scaled, 
        y_train, 
        y_test, 
        description, 
        svr
        )

# Trying different polynomials
#degrees = [3,4,5,6,7,8,9,10,11,12]
#degrees = [11,12,13,14,15,16,17,18,19,20]
degrees = [3] # list of 1 to make it quicker for Jacob / Atif to run my script
kernel = "poly"
for degree in degrees:
    description = "Support vector regression, unscaled data. Kernel=" + kernel
    svr = SVR(kernel, degree=degree) # default kernel='RBF'
    _ = train_test_regressor(x_train, x_test, y_train, y_test, description, svr)
    
# Random search SVR
svr = SVR()
distributions = dict(C=uniform(loc=0, scale=4), 
                     epsilon=uniform(loc=0, scale=1),
                     kernel=["linear", "poly", "rbf", "sigmoid"])
rscv = RandomizedSearchCV(svr, distributions, n_iter=random_search_n_svr)
search = rscv.fit(x_train, y_train) # experiments above showed unscaled data was best
print("Best parameters: \n {}".format(search.best_params_))

svr = SVR(kernel=search.best_params_["kernel"],
          C=search.best_params_["C"],
          epsilon=search.best_params_["epsilon"])

description = "SVR random search"

cross_validate(df, svr, target_variable)

########## Plotting the performance of different C and epislon values #########
c_params = [1,3,5,7,10, 20, 40, 60, 100, 300, 600, 1000, 2000, 10000, 20000]
description = "SVR with different C values"
scores = []
for param in c_params:
    svr = SVR(C=param)
    
    trained_model, rmse = train_test_regressor(x_train, x_test, y_train, y_test, description, svr)
    scores.append(rmse)
    
plt.plot(c_params, scores)
plt.ylabel("RMSE")
plt.xlabel("C")
title = "SVR RMSE as C increases, range " + str(np.min(c_params)) + " - " + str(np.max(c_params))
plt.title(title)
plt.show()

epsilons = [0.9, 0.5, 0.1, 0.01, 0.0001]
description = "SVR with different epsilon values"
scores = []
for param in epsilons:
    svr = SVR(epsilon=param)
    
    trained_model, rmse = train_test_regressor(x_train, x_test, y_train, y_test, description, svr)
    scores.append(rmse)
    
plt.plot(epsilons, scores)
plt.ylabel("RMSE")
plt.xlabel("Epsilon")
title = "SVR RMSE as epsilon decreases, range " + str(np.min(epsilons)) + " - " + str(np.max(epsilons))
plt.title(title)
plt.show()

###############################################################################
############################ Elastic net ######################################
###############################################################################

elastic_net = ElasticNet(alpha=0.00001, l1_ratio=0.2) # alpha=0.1, l1_ratio=0.5
description = "Elastic net, no data scaling"
_ = train_test_regressor(x_train, x_test, y_train, y_test, description, elastic_net)

# Random search elastic net
distributions = dict(alpha=uniform(loc=0, scale=1), l1_ratio=uniform(loc=0, scale=1))
rscv = RandomizedSearchCV(elastic_net, distributions, n_iter=random_seach_n)
search = rscv.fit(x_train, y_train) # experiments above showed unscaled data was best
search.best_params_

elastic_net = ElasticNet(
    alpha=search.best_params_["alpha"],
    l1_ratio=search.best_params_["l1_ratio"]) # alpha=0.1, l1_ratio=0.5
description = "Elastic net, no data scaling"
#_ = train_test_regressor(x_train, x_test, y_train, y_test, description, elastic_net)

cross_validate(df, elastic_net, target_variable)

###############################################################################
############################ Random forest ####################################
###############################################################################

forest = RandomForestRegressor(max_features=12, n_estimators=1000)
description = "Random forest regression"
_ = train_test_regressor(x_train, x_test, y_train, y_test, description, forest)

# Random forest random search optimization - adapted from Koehrsen (2018)
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt', 'log2']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf_random = RandomizedSearchCV(
    estimator = forest, 
    param_distributions = random_grid, 
    n_iter = random_seach_n, 
    cv = 3, 
    verbose=2, 
    random_state=42, 
    n_jobs = -1
    )

rf_random.fit(x_train, y_train)

print("Random Forest best parameters : \n {}".format(rf_random.best_params_))

cross_validate(df, rf_random.best_estimator_, target_variable)

###############################################################################
#################### Gradient booster regression ##############################
###############################################################################
description = "Gradient Boosting Regression"
reg = GradientBoostingRegressor(n_estimators=1000, max_depth=2, random_state=0, learning_rate=0.1)
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt', 'log2']
max_depth = [int(x) for x in range(1, 30, 3)]
max_depth.append(None)
min_samples_split = [2, 3, 5, 8, 13]
min_samples_leaf = [1, 2, 4]
learning_rate = [0.0001, 0.001, 0.01, 0.1, 1]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'learning_rate' : learning_rate}

search = RandomizedSearchCV(
    estimator = reg, 
    param_distributions = random_grid, 
    n_iter = random_seach_n, 
    cv = 3, 
    verbose=2, 
    random_state=42, 
    n_jobs = -1
    )

search.fit(x_train, y_train)

print("Gradient boosting best parameters : \n {}".format(search.best_params_))

cross_validate(df, search.best_estimator_, target_variable)

# Running new random search with values close to the best params identified above
best_params = {'n_estimators': 1000, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 7}
n_estimators = [int(x) for x in np.linspace(start = 800, stop = 1200, num = 10)]
max_features = ['log2']
max_depth = [int(x) for x in range(3, 14)]
max_depth.append(None)
min_samples_split = [2, 3, 5]
min_samples_leaf = [1, 2, 3, 4]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

search = RandomizedSearchCV(
    estimator = reg, 
    param_distributions = random_grid, 
    n_iter = random_seach_n, 
    cv = 3, 
    verbose=2, 
    random_state=42, 
    n_jobs = -1
    )

search.fit(x_train, y_train)

print("Gradient boosting best parameters : \n {}".format(search.best_params_))

cross_validate(df, search.best_estimator_, target_variable)

#_ = train_test_regressor(x_train, x_test, y_train, y_test, description, search.best_estimator_)


############### Gradient booster: Plotting effect of hyperparameters ##########

## Effect of n_estimators
scores = []
#n_estimators = [n for n in range(10, 1500, 30)]
params = [n for n in range(10, 500, 10)]
for n in [750, 1000, 1500]:
    params.append(n)
for n in params:
    regressor = GradientBoostingRegressor(n_estimators=n)
    trained_model, rmse = train_test_regressor(x_train, x_test, y_train, y_test, description, regressor)
    scores.append(rmse)
    
plt.plot(params, scores)
plt.ylabel("RMSE")
plt.xlabel("n_estimators")
title = "Gradient booster RMSE as n_estimators increase, range " + str(np.min(params)) + " - " + str(np.max(params))
plt.title(title)
plt.show()

idx = scores.index(np.min(scores))
best_n_estimators = params[idx]
print("n_estimator with lowest RMSE is: {}".format(best_n_estimators))
print("Lowest RMSE: {}".format(scores[idx]))

# Running loops again, but this time with values close to the best value from previous round
scores = []
params = [n for n in range(best_n_estimators-10, best_n_estimators+10, 2)]
for n in params:
    regressor = GradientBoostingRegressor(n_estimators=n)
    trained_model, rmse = train_test_regressor(x_train, x_test, y_train, y_test, description, regressor)
    scores.append(rmse)
    
plt.plot(params, scores)
plt.ylabel("RMSE")
plt.xlabel("n_estimators")
title = "Gradient boost RMSE as n_estimators increase, range " + str(np.min(params)) + " - " + str(np.max(params))
plt.title(title)
plt.show()

idx = scores.index(np.min(scores))
best_n_estimators = params[idx]
print("n_estimator with lowest RMSE is: {}".format(best_n_estimators))  
print("Lowest RMSE: {}".format(scores[idx]))

## Effect of learning rate
scores = []
params = [0.0001, 0.001, 0.01, 0.1, 1]

for n in params:
    regressor = GradientBoostingRegressor(learning_rate=n)
    trained_model, rmse = train_test_regressor(x_train, x_test, y_train, y_test, description, regressor)
    scores.append(rmse)
    
plt.plot(params, scores)
plt.ylabel("RMSE")
plt.xlabel("learning rate")
title = "Gradient boost RMSE as learning rate increases, range " + str(np.min(params)) + " - " + str(np.max(params))
plt.title(title)
plt.show()

idx = scores.index(np.min(scores))
best_lr = params[idx]
print("Learning rate with lowest RMSE is: {}".format(best_lr))
print("Lowest RMSE: {}".format(scores[idx]))

## Effect of max_features
params = ['auto', 'sqrt', 'log2']
scores = []

for n in params:
    regressor = GradientBoostingRegressor(max_features=n)
    trained_model, rmse = train_test_regressor(x_train, x_test, y_train, y_test, description, regressor)
    scores.append(rmse)
    
plt.bar(params, scores)
plt.ylabel("RMSE")
plt.xlabel("max_feature")
title = "Gradient boost RMSE for different max_features"
plt.title(title)
plt.show()

idx = scores.index(np.min(scores))
best_max_features = params[idx]
print("max_features with lowest RMSE is: {}".format(best_lr))
print("Lowest RMSE: {}".format(scores[idx]))

## Effect of max_depth
scores = []
params = [int(x) for x in range(1, 30, 3)]

for n in params:
    regressor = GradientBoostingRegressor(max_depth=n)
    trained_model, rmse = train_test_regressor(x_train, x_test, y_train, y_test, description, regressor)
    scores.append(rmse)
    
plt.plot(params, scores)
plt.ylabel("RMSE")
plt.xlabel("max_depth")
title = "Gradient boost RMSE as max depth increases, range " + str(np.min(params)) + " - " + str(np.max(params))
plt.title(title)
plt.show()

idx = scores.index(np.min(scores))
best_max_depth = params[idx]
print("max_depth with lowest RMSE is: {}".format(best_max_depth))
print("Lowest RMSE: {}".format(scores[idx]))

## Effect of min_samples_split
params = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
scores = []

for n in params:
    regressor = GradientBoostingRegressor(min_samples_split=n)
    trained_model, rmse = train_test_regressor(x_train, x_test, y_train, y_test, description, regressor)
    scores.append(rmse)
    
plt.plot(params, scores)
plt.ylabel("RMSE")
plt.xlabel("min_samples_split")
title = "Gradient boost RMSE as min_samples_split increases, range " + str(np.min(params)) + " - " + str(np.max(params))
plt.title(title)
plt.show()

idx = scores.index(np.min(scores))
best_min_samples_split = params[idx]
print("min_samples_split with lowest RMSE is: {}".format(best_max_depth))
print("Lowest RMSE: {}".format(scores[idx]))

## Effect of min_samples_leaf 
params = [1, 2, 4, 6, 8, 10, 12, 14, 28, 56, 112, 224]
scores = []

for n in params:
    regressor = GradientBoostingRegressor(min_samples_leaf=n)
    trained_model, rmse = train_test_regressor(x_train, x_test, y_train, y_test, description, regressor)
    scores.append(rmse)
    
plt.plot(params, scores)
plt.ylabel("RMSE")
plt.xlabel("min_samples_split")
title = "Gradient boost RMSE as min_samples_leaf increases, range " + str(np.min(params)) + " - " + str(np.max(params))
plt.title(title)
plt.show()

idx = scores.index(np.min(scores))
best_min_samples_leaf = params[idx]
print("min_samples_leaf with lowest RMSE is: {}".format(best_max_depth))
print("Lowest RMSE: {}".format(scores[idx]))

# Running again with smaller range
params = [1, 2, 3, 4, 5, 6]
scores = []

for n in params:
    regressor = GradientBoostingRegressor(min_samples_leaf=n)
    trained_model, rmse = train_test_regressor(x_train, x_test, y_train, y_test, description, regressor)
    scores.append(rmse)
    
plt.plot(params, scores)
plt.ylabel("RMSE")
plt.xlabel("min_samples_split")
title = "Gradient boost RMSE as min_samples_leaf increases, range " + str(np.min(params)) + " - " + str(np.max(params))
plt.title(title)
plt.show()

idx = scores.index(np.min(scores))
min_samples_leaf = params[idx]
print("min_samples_leaf with lowest RMSE is: {}".format(best_max_depth))
print("Lowest RMSE: {}".format(scores[idx]))

## Running gradient booster with best params identified above
description = "Gradient booster with best hyperparams from manual process"
regressor = GradientBoostingRegressor(min_samples_leaf=best_min_samples_leaf,
                                      n_estimators=best_n_estimators,
                                      min_samples_split=best_min_samples_split,
                                      max_depth=best_max_depth,
                                      max_features=best_max_features
                                      )

#_ = train_test_regressor(x_train, x_test, y_train, y_test, description, regressor)

## Evaluating model with best hyperparameters identified above
cross_validate(df, regressor, target_variable)


###############################################################################
################################### XGBoost ###################################
###############################################################################

# Running random search
regressor = XGBRegressor()
# Fitting and evaluating model
description = "XGBoost random search"

# Parameters to tune
eta = [0.1, 0.01, 0.001, 0.0001]
max_depth = [int(x) for x in range(1, 30, 3)]
subsample = [0.5, 0.75, 1]
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 5000, num = 10)]
colsample_bytree = np.arange(0.1, 1.1, 0.1)


random_grid = {'eta': eta,
               'max_depth': max_depth,
               'subsample': subsample,
               'n_estimators': n_estimators,
               'colsample_bytree': colsample_bytree}
search = RandomizedSearchCV(
    estimator = regressor, 
    param_distributions = random_grid, 
    n_iter = random_seach_n, 
    cv = 3, 
    verbose=2, 
    random_state=42, 
    n_jobs = -1
    )

search.fit(x_train, y_train)
print("XGBoost best parameters : \n {}".format(search.best_params_))
_ = train_test_regressor(x_train, x_test, y_train, y_test, description, search.best_estimator_)

# Cross validation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

n_scores = cross_val_score(regressor, 
                           x_train, 
                           y_train,
                           scoring="neg_root_mean_squared_error",
                           cv=cv, 
                           n_jobs=-1, 
                           error_score='raise')

print("Mean negative RMSE from 10 fold CV: \n {}".format(np.mean(n_scores)))

################ XGBoost: Plotting effect of hyperparameters ##################

## Plotting results against n_estimators
scores = []
#n_estimators = [n for n in range(10, 1500, 30)]
params = [n for n in range(10, 500, 10)]
for n in [750, 1000, 1500]:
    params.append(n)
for n in params:
    regressor = XGBRegressor(n_estimators=n)
    trained_model, rmse = train_test_regressor(x_train, x_test, y_train, y_test, description, regressor)
    scores.append(rmse)
    
plt.plot(params, scores)
plt.ylabel("RMSE")
plt.xlabel("n_estimators")
title = "XGBoost RMSE as n_estimators increase, range " + str(np.min(params)) + " - " + str(np.max(params))
plt.title(title)
plt.show()

idx = scores.index(np.min(scores))
best_n_estimators = params[idx]
print("n_estimator with lowest RMSE is: {}".format(best_n_estimators))
print("Lowest RMSE: {}".format(scores[idx]))

# Running loops again, but this time with values close to the best value from previous round
scores = []
#n_estimators = [n for n in range(10, 1500, 30)]
params = [n for n in range(best_n_estimators-10, best_n_estimators+10, 2)]
for n in params:
    regressor = XGBRegressor(n_estimators=n)
    trained_model, rmse = train_test_regressor(x_train, x_test, y_train, y_test, description, regressor)
    scores.append(rmse)
    
plt.plot(params, scores)
plt.ylabel("RMSE")
plt.xlabel("n_estimators")
title = "XGBoost RMSE as n_estimators increase, range " + str(np.min(params)) + " - " + str(np.max(params))
plt.title(title)
plt.show()

idx = scores.index(np.min(scores))
best_n_estimators = params[idx]
print("n_estimator with lowest RMSE is: {}".format(best_n_estimators))  
print("Lowest RMSE: {}".format(scores[idx]))


## Plotting results against eta (learning rate)
scores = []
params = [0.0001, 0.001, 0.01, 0.1, 1]

for n in params:
    regressor = XGBRegressor(eta=n)
    trained_model, rmse = train_test_regressor(x_train, x_test, y_train, y_test, description, regressor)
    scores.append(rmse)
    
plt.plot(params, scores)
plt.ylabel("RMSE")
plt.xlabel("eta (learning rate)")
title = "XGBoost RMSE as learning rate increases, range " + str(np.min(params)) + " - " + str(np.max(params))
plt.title(title)
plt.show()

idx = scores.index(np.min(scores))
best_lr = params[idx]
print("Learning rate with lowest RMSE is: {}".format(best_lr))
print("Lowest RMSE: {}".format(scores[idx]))


# Plotting results against max depth
scores = []
params = [int(x) for x in range(1, 30, 3)]

for n in params:
    regressor = XGBRegressor(max_depth=n)
    trained_model, rmse = train_test_regressor(x_train, x_test, y_train, y_test, description, regressor)
    scores.append(rmse)
    
plt.plot(params, scores)
plt.ylabel("RMSE")
plt.xlabel("max_depth")
title = "XGBoost RMSE as max depth increases, range " + str(np.min(params)) + " - " + str(np.max(params))
plt.title(title)
plt.show()

idx = scores.index(np.min(scores))
best_max_depth = params[idx]
print("max_depth with lowest RMSE is: {}".format(best_max_depth))
print("Lowest RMSE: {}".format(scores[idx]))

scores = []
params = [int(x) for x in range(best_max_depth//2, best_max_depth*2, 1)]
for n in params:
    regressor = XGBRegressor(max_depth=n)
    trained_model, rmse = train_test_regressor(x_train, x_test, y_train, y_test, description, regressor)
    scores.append(rmse)
    
plt.plot(params, scores)
plt.ylabel("RMSE")
plt.xlabel("max_depth")
title = "XGBoost RMSE as max depth increases, range " + str(np.min(params)) + " - " + str(np.max(params))
plt.title(title)
plt.show()

idx = scores.index(np.min(scores))
best_max_depth = params[idx]
print("max_depth with lowest RMSE is: {}".format(best_max_depth))
print("Lowest RMSE: {}".format(scores[idx]))

## Plotting results against subsample ratio
scores = []
params = [0.5, 0.75, 1]

for n in params:
    regressor = XGBRegressor(subsample=n)
    trained_model, rmse = train_test_regressor(x_train, x_test, y_train, y_test, description, regressor)
    scores.append(rmse)
    
plt.plot(params, scores)
plt.ylabel("RMSE")
plt.xlabel("subsample")
title = "XGBoost RMSE as subsample increases, range " + str(np.min(params)) + " - " + str(np.max(params))
plt.title(title)
plt.show()

idx = scores.index(np.min(scores))
best_subsample = params[idx]
print("subsample with lowest RMSE is: {}".format(best_subsample))
print("Lowest RMSE: {}".format(scores[idx]))

## Plotting results against colsample
params = np.arange(0.1, 1.1, 0.1)
scores = []

for n in params:
    regressor = XGBRegressor(colsample_bytree=n)
    trained_model, rmse = train_test_regressor(x_train, x_test, y_train, y_test, description, regressor)
    scores.append(rmse)
    
plt.plot(params, scores)
plt.ylabel("RMSE")
plt.xlabel("colsample_bytree")
title = "XGBoost RMSE as colsample_bytree increases, range " + str(np.min(params)) + " - " + str(np.max(params))
plt.title(title)
plt.show()

idx = scores.index(np.min(scores))
best_colsample_bytree = params[idx]
print("colsample_bytree with lowest RMSE is: {}".format(best_colsample_bytree))
print("Lowest RMSE: {}".format(scores[idx]))

description = "XGBoost best params"
xgboost_best_params = XGBRegressor(n_estimators=best_n_estimators,
                                   subsample=best_subsample,
                                   eta=best_lr,
                                   colsample_bytree=best_colsample_bytree,
                                   max_depth=best_max_depth
                                   )

## Evaluating model with best hyperparameters identified above
cross_validate(df, xgboost_best_params, target_variable)
















