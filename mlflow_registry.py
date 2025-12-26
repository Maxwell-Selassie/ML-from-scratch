
import mlflow
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

mlflow.set_experiment('Model Registry Tutorial')

with mlflow.start_run(run_name='RegressionModel-ModelRegistry'):
    X, y = make_regression(n_features=4, n_informative=2, random_state=1, shuffle=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    params = {'max_depth': 2, 'random_state': 1}
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    mlflow.log_params(params)

    prediction = model.predict(X_test)
    mlflow.log_metrics({'mse' : mean_squared_error(y_test, prediction)})

    # log the sklearn model and register as version 1
    mlflow.sklearn.log_model(
        sk_model=model,
        name='sklearn_model',
        input_example=X_train,
        registered_model_name='sklearn-randomforest-reg-model'
    )
