from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def get_logistic_model():
    return LogisticRegression(random_state=42, max_iter=1000)

def get_svm_model():
    return SVC(random_state=42, kernel='rbf')

def get_rf_model():
    return RandomForestClassifier(random_state=42, n_estimators=100)
