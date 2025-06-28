# Hyperparameter Tuning

### Topic: Grid Search, Random Search, and Bayesian Optimization

---

##  Summary

* Understand the **difference between parameters and hyperparameters**
* Learn **Grid Search** and **Randomized Search** using `sklearn`
* Dive into **Bayesian Optimization** using `optuna` or `hyperopt`
* Practice **hyperparameter tuning** on models like **Random Forest** and **XGBoost**

---

## 1. Parameters vs Hyperparameters

| Type                 | Description                                                            | Set By         |
| -------------------- | ---------------------------------------------------------------------- | -------------- |
| **Model Parameters** | Learned from data during training (e.g., weights in linear regression) | The algorithm  |
| **Hyperparameters**  | Set **before** training (e.g., tree depth, learning rate)              | You (the user) |

---

###  Analogy: Baking a Cake

> * **Ingredients like sugar, flour, eggs** are **hyperparameters** — you choose them **before** baking.
> * **How the cake rises, texture, and taste** are like model parameters — these emerge **during baking**.

If the hyperparameters are wrong (e.g., too much flour), the cake won’t rise well — just like a poorly tuned model!

---

## 2. Why Hyperparameter Tuning Matters

Hyperparameters control how well your model:

* Learns patterns (e.g., `learning_rate`)
* Generalizes to unseen data (e.g., `max_depth`, `min_samples_split`)
* Trades off bias vs variance

---

## 3. Grid Search (Exhaustive)

**Grid Search** tries every possible combination of hyperparameters from a predefined set.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load data
X, y = load_iris(return_X_y=True)

# Define model
model = RandomForestClassifier()

# Define grid
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [2, 4, 6]
}

# Grid search
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X, y)

print("Best Params:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
```

---

###  Analogy: Grid Search = Trying Every Pizza

> You want to find your favorite pizza by trying **every combination** of crust, toppings, and cheese.
> It’s thorough but **takes time** (especially when you have many options).

---

## 4. Randomized Search

**Randomized Search** randomly samples from the hyperparameter space. It doesn't test all combinations, but it's **faster** and often **just as good**.

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

# Define model
model = RandomForestClassifier()

# Define distributions
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(2, 10)
}

# Random search
random_search = RandomizedSearchCV(model, param_distributions=param_dist,
                                   n_iter=10, cv=5, random_state=42)
random_search.fit(X, y)

print("Best Params:", random_search.best_params_)
print("Best Score:", random_search.best_score_)
```

---

###  Analogy: Randomized Search = Taste Testing Samples

> Instead of eating every pizza, you try **10 random ones**.
> You may not find the best **ever**, but you’ll find a **very good one** with much less effort.

---

## 5. Bayesian Optimization (with Optuna)

Unlike Grid or Random Search, **Bayesian Optimization** uses **previous results** to decide what to try next. It’s **smarter** and **more efficient**.

###  How It Works:

1. Start with a few random samples
2. Fit a probabilistic model to the results
3. Predict which set of hyperparameters will perform best next
4. Repeat

---

###  Using Optuna

```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 2, 10)

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    score = cross_val_score(model, X, y, cv=5).mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print("Best Params:", study.best_params)
print("Best Score:", study.best_value)
```

---

###  Analogy: Smart Chef Tuning Recipes

> The chef tries a recipe, adjusts based on taste, and learns **what changes to make next**.
> This is smarter than randomly guessing (Random Search) or exhaustively trying everything (Grid Search).

---

## 6. Tuning XGBoost (Real-World Example)

```python
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_breast_cancer
from scipy.stats import uniform

X, y = load_breast_cancer(return_X_y=True)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': uniform(0.01, 0.3),
    'max_depth': [3, 5, 7],
    'subsample': uniform(0.5, 0.5)
}

search = RandomizedSearchCV(model, param_distributions=param_dist,
                            n_iter=10, scoring='accuracy', cv=5, random_state=42)
search.fit(X, y)

print("Best Parameters:", search.best_params_)
print("Best Accuracy:", search.best_score_)
```

---

## 7. When to Use What?

| Method                         | Use When                                                       |
| ------------------------------ | -------------------------------------------------------------- |
| **Grid Search**                | You have few hyperparameters and compute power isn’t a concern |
| **Random Search**              | Large search space, limited time                               |
| **Bayesian (Optuna/Hyperopt)** | You want efficiency and smart exploration                      |

---

## 8. Final Analogy Recap

| Analogy          | Concept                |
| ---------------- | ---------------------- |
| Cake Ingredients | Hyperparameters        |
| Pizza Sampling   | Grid and Random Search |
| Smart Chef       | Bayesian Optimization  |

---

## 9. Summary Table

| Topic                         | Tool/Concept           | Use                    |
| ----------------------------- | ---------------------- | ---------------------- |
| Parameters vs Hyperparameters | Core ML concept        | Setup vs learned       |
| Grid Search                   | `GridSearchCV`         | Exhaustive testing     |
| Random Search                 | `RandomizedSearchCV`   | Fast + flexible        |
| Bayesian Optimization         | `optuna`, `hyperopt`   | Smart, adaptive tuning |
| Real-World Models             | Random Forest, XGBoost | Hands-on tuning        |

---


