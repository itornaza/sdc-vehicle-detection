
from sklearn.svm import LinearSVC
import time

def _check_predictions(svc, X_test, y_test):
    '''Check the accuracy and sample prediction of the classifier'''
    
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')


def classify(X_train, X_test, y_train, y_test, vis=False):
    '''Create and train a linear svc'''
    
    # Use a linear SVC
    svc = LinearSVC()
    
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    
    if vis:
        print(round(t2-t, 2), 'Seconds to train SVC...')
        _check_predictions(svc, X_test, y_test)

    return svc
