from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import LabelEncoder

def balance_data(df):
    data = df.values
    # split into input and output elements
    X_train, y_train = data[:, :-1], data[:, -1]
    # label encode the target variable
    y_train = LabelEncoder().fit_transform(y_train)
    # summarize distribution
    counter = Counter(y_train)
    print("Before balancing:")
    for k,v in counter.items():
        per = v / len(y_train) * 100
        print('Class=%d, n=%d (%.3f%%)' % (k, v, per))

    # Balance the dataset with respect to stances
    y_train = LabelEncoder().fit_transform(y_train)
    # transform the dataset
    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)
    # summarize distribution
    counter = Counter(y_train)
    print("After balancing:")
    for k,v in counter.items():
        per = v / len(y_train) * 100
        print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
    # map 0, 1, 2 to -1, 0, 1
    y_train = y_train - 1
    return X_train, y_train