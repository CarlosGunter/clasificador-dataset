import numpy as np

atributos = np.genfromtxt('assets/TUANDROMD.csv', delimiter=',', dtype=int, encoding=None, skip_header=1, usecols=range(0, 241))
tipo = np.genfromtxt('assets/TUANDROMD.csv', delimiter=',', dtype=str, encoding=None, skip_header=1, usecols=-1)
tipo_int = np.array([1 if x == 'malware' else 0 for x in tipo])

def SVC(atributos, tipo):
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test = train_test_split(atributos, tipo, test_size=0.2, random_state=0)

    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    return accuracy_score(y_test, y_pred)

print(SVC(atributos, tipo_int))