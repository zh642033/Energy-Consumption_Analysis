import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb

sns.set_theme(style='darkgrid', palette='colorblind')
df = pd.read_csv('global_energy_consumption.csv')

target = 'Primary energy consumption per capita (kWh/person)'
features = [
    'Access to electricity (% of population)',
    'GDP per capita','Density (P/km2)',
    'Financial flows to developing countries (US $)',
    'Renewable electricity Generating Capacity per capita',
    'Electricity from fossil fuels (TWh)',
    'Electricity from nuclear (TWh)','Electricity from renewables (TWh)','Low-carbon electricity (% electricity)',
    'Renewables (% equivalent primary energy)']
df_drop_missing_target= df.dropna(subset=[target])
x = df_drop_missing_target[features]
y = df_drop_missing_target[target]
imputer = SimpleImputer(strategy='mean')
x_imputed = imputer.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(x_imputed, y, test_size = 0.2, random_state = 42)

# scaler = StandardScaler()
# x_train_scaled = scaler.fit_transform(x_train)
# x_test_scaled = scaler.transform(x_test)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.feature_selection import SelectKBest


class FeatureEngineering:
    def __init__(self, poly_degree=2):
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=poly_degree, interaction_only=False, include_bias=False)
        self.variance_threshold = VarianceThreshold(threshold=0.1)

    def fit_transform(self, X_train):
        """Apply scaling, polynomial feature generation, and variance threshold on training data."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_poly = self.poly.fit_transform(X_train_scaled)
        X_train_selected = self.variance_threshold.fit_transform(X_train_poly)
        return X_train_selected

    def transform(self, X_test):
        """Apply transformations on test data based on training data's fit."""
        X_test_scaled = self.scaler.transform(X_test)
        X_test_poly = self.poly.transform(X_test_scaled)
        X_test_selected = self.variance_threshold.transform(X_test_poly)
        return X_test_selected


class FeatureSelection:
    def __init__(self, method='mutual_info', k=10):
        self.method = method
        self.k = k

    def select_features(self, X_train, y_train):
        """Select important features using the specified method."""
        if self.method == 'mutual_info':
            selector = SelectKBest(mutual_info_regression, k=self.k)
        else:
            raise ValueError(f"Unsupported selection method: {self.method}")
        
        return selector.fit_transform(X_train, y_train), selector

    def transform(self, X_test, selector):
        """Apply the same feature selection on the test set."""
        return selector.transform(X_test)


class HyperparameterTuning:
    def __init__(self, regressor_type):
        self.param_grids = {
            'ridge': {'alpha': [0.1, 1.0, 10.0]},
            'lasso': {'alpha': [0.01, 0.1, 1.0]},
            'random_forest': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
            'svr': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
            'kneighbors': {'n_neighbors': [3, 5, 7]},
            'gbr': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
            'adaboost': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
            'extra_trees': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
            'lightgbm': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
        }
        self.regressor_type = regressor_type
        self.param_grid = self.param_grids.get(regressor_type, {})

    def tune_model(self, model, X_train, y_train):
        """Tune the model's hyperparameters using GridSearchCV."""
        if not self.param_grid:
            print(f"No hyperparameters to tune for {self.regressor_type}")
            return model
        grid_search = GridSearchCV(estimator=model, param_grid=self.param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Best Parameters for {self.regressor_type}: {grid_search.best_params_}")
        return best_model


class RegressionModelComparison:
    def __init__(self, regressor_type='linear'):
        self.regressor_type = regressor_type
        self.model = self._select_regressor()

    def _select_regressor(self):
        """Select the regressor based on the type."""
        if self.regressor_type == 'linear':
            return LinearRegression()
        elif self.regressor_type == 'ridge':
            return Ridge()
        elif self.regressor_type == 'lasso':
            return Lasso()
        elif self.regressor_type == 'random_forest':
            return RandomForestRegressor()
        elif self.regressor_type == 'svr':
            return SVR()
        elif self.regressor_type == 'kneighbors':
            return KNeighborsRegressor()
        elif self.regressor_type == 'gbr':
            return GradientBoostingRegressor()
        elif self.regressor_type == 'adaboost':
            return AdaBoostRegressor()
        elif self.regressor_type == 'extra_trees':
            return ExtraTreesRegressor()
        elif self.regressor_type == 'lightgbm':
            return lgb.LGBMRegressor()
        else:
            raise ValueError(f"Unsupported regressor type: {self.regressor_type}")

    def train(self, X_train, y_train):
        """Train the selected model."""
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Evaluate the model and return MSE and R-squared."""
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2

    def run(self, X_train, y_train, X_test, y_test, feature_engineer=None, feature_selector=None, hyperparameter_tuner=None):
        """Apply feature engineering, feature selection, hyperparameter tuning, train and evaluate the model."""
        if feature_engineer:
            X_train = feature_engineer.fit_transform(X_train)
            X_test = feature_engineer.transform(X_test)
        
        if feature_selector:
            X_train, selector = feature_selector.select_features(X_train, y_train)
            X_test = feature_selector.transform(X_test, selector)

        if hyperparameter_tuner:
            self.model = hyperparameter_tuner.tune_model(self.model, X_train, y_train)

        self.train(X_train, y_train)
        mse, r2 = self.evaluate(X_test, y_test)
        return mse, r2


if __name__ == "__main__":
  

    feature_engineer = FeatureEngineering(poly_degree=2)
    feature_selector = FeatureSelection(method='mutual_info', k=10)

    regressors = ['ridge', 'lasso', 'random_forest', 'svr', 'kneighbors', 'gbr', 'adaboost', 'extra_trees', 'lightgbm']

    results = []

    for regressor in regressors:
        print(f"Running model: {regressor}")
        
        model = RegressionModelComparison(regressor_type=regressor)

        # Step 1: Evaluate original model performance
        mse_original, r2_original = model.run(X_train, y_train, X_test, y_test)
        
        # Step 2: Evaluate performance with feature engineering
        mse_feat_eng, r2_feat_eng = model.run(X_train, y_train, X_test, y_test, feature_engineer=feature_engineer, feature_selector=feature_selector)
        
        # Step 3: Hyperparameter tuning & model selection
        hyperparameter_tuner = HyperparameterTuning(regressor_type=regressor)
        mse_tuned, r2_tuned = model.run(X_train, y_train, X_test, y_test, feature_engineer=feature_engineer, feature_selector=feature_selector, hyperparameter_tuner=hyperparameter_tuner)
        
        results.append({
            'Model': regressor,
            'MSE_Original': mse_original,
            'R2_Original': r2_original,
            'MSE_Feature_Eng': mse_feat_eng,
            'R2_Feature_Eng': r2_feat_eng,
            'MSE_Tuned': mse_tuned,
            'R2_Tuned': r2_tuned
        })

    results_df = pd.DataFrame(results)
    print("\nComparison of Model Performance:")

    results_df =results_df.sort_values(by=['R2_Original'])
    print(results_df)

    results_df.to_csv('resutls.csv')