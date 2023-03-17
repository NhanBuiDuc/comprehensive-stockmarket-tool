from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from config import config as cf
from sklearn.model_selection import train_test_split
import infer
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import utils
def train_random_forest_classfier(X_train, X_test, X_val, y_train, y_test, y_val):
    day_steps = cf["model"]["rdfc"]["output_dates"]
    y_train = [-1 if y_train[i] > y_train[i+day_steps] else 1 for i in range(len(y_train) - day_steps)]
    y_test = [-1 if y_test[i] > y_test[i+day_steps] else 1 for i in range(len(y_test) - day_steps)]
    y_val = [-1 if y_val[i] > y_val[i+day_steps] else 1 for i in range(len(y_val) - day_steps)]

    # train a random forest classifier using scikit-learn
    model = RandomForestClassifier(n_estimators=1000, random_state=42)
    model.fit(X_train[:-day_steps], y_train)
    y_pred = model.predict(X_val[:-day_steps])

    # calculate the accuracy of the predictions
    accuracy = accuracy_score(y_val, y_pred)
    print("Val Accuracy:", accuracy)
    # Print the F1 score
    f1 = f1_score(y_val, y_pred)
    print("Val F1 score: {:.2f}".format(f1))
    print("Weights:", model.feature_importances_)
    random_forest_test_acc, test_f1 = infer.test_random_forest_classfier(model, X_test[:-day_steps], y_test)
    print("Test Accuracy:", random_forest_test_acc)
    # Print the F1 score
    print("Test F1 score: {:.2f}".format(test_f1))


    return model


def train_random_forest_regressior(X_train, y_train):
    y_train = np.array(y_train)

    y_train = (utils.diff(y_train))

    # Create a Random Forest Regression model with 100 trees
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model on the training data
    model.fit(X_train[:-1], y_train)
    return model
