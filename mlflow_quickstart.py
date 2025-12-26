
import mlflow

from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

mlflow.set_experiment('Hyperparameter_Tuning')

digits = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, stratify=digits.target, random_state=1
)

mlflow.sklearn.autolog(max_tuning_runs=10)

params_grid = {
    'n_estimators' : [100,150,200],
    'max_depth' : [3, 5, 7, None],
    'min_samples_split' : [2, 5, 10]
}

with mlflow.start_run(run_name='RF Hyperparameter Tuning'):
    rfc = RandomForestClassifier(random_state=2)
    grid_search = GridSearchCV(rfc, params_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # evaluation metrics
    best_score = grid_search.score(X_test, y_test)
    print(f'Best params: {grid_search.best_params_}')
    print(f'Best CV score: {grid_search.best_score_:.3f}')
    print(f'Test score: {best_score:.2f}')