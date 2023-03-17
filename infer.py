from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def test_random_forest_classfier(model, X_test, y_test):

    y_pred = model.predict(X_test)

    # calculate the accuracy of the predictions
    accuracy = accuracy_score(y_test, y_pred)
    # Calculate the F1 score of the model's predictions
    f1 = f1_score(y_test, y_pred)

    return accuracy, f1