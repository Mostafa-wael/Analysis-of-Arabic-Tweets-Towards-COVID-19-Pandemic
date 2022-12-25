# For models
from sklearn.metrics import classification_report
import pickle


def trainModel(X_train, y_train, model):
    model.fit(X_train, y_train)
    return model

def testModel(X_test, model):
    y_pred = model.predict(X_test)
    return y_pred

def evaluateModel(y_test, y_pred):  
    print(classification_report(y_test, y_pred))
    return classification_report(y_test, y_pred, output_dict=True)['accuracy']

def saveModel(model, model_name):
    pickle.dump(model, open(model_name, 'wb'))
    return model_name

def loadModel(model_name):
    return pickle.load(open(model_name, 'rb'))

def modelPipeline(X_train, y_train, X_test, y_test, model, model_name):
    model = trainModel(X_train, y_train, model)
    y_pred = testModel(X_test, model)
    report = evaluateModel(y_test, y_pred)
    saveModel(model, model_name)
    return model, report