from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

RANDOM_STATE = 42


def build_model_lgr():

    base_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="liblinear"
    )

    model_lgr = OneVsRestClassifier(base_model)

    return model_lgr 


def build_model_svc():

    base_model = LinearSVC(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=10000
    )

    model_svc = OneVsRestClassifier(base_model)

    return model_svc