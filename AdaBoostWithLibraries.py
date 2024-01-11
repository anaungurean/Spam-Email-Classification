import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

class AdaBoostWithLibraries:
    def __init__(self, file_name='train_data_processed.csv', test_file_name='test_data_processed.csv', n_estimators=50):
        self.train_file_name = file_name
        self.test_file_name = test_file_name
        self.train_data = self.load_data(self.train_file_name)
        self.test_data = self.load_data(self.test_file_name)
        self.vectorizer = CountVectorizer()
        self.n_estimators = n_estimators
        self.classifier = AdaBoostClassifier(n_estimators=self.n_estimators)
        self.accuracy = None

    def load_data(self, file_name=None):
        data = pd.read_csv(file_name)
        return data

    def preprocess_data(self):
        X = self.vectorizer.fit_transform(self.train_data['text'])
        y = self.train_data['is_spam'].astype('int')
        return X, y

    def train(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)

    def predict(self, X_test):
        return self.classifier.predict(X_test)

    def evaluate_on_train_data(self):
        X, y = self.preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.train(X_train, y_train)
        y_pred_train = self.predict(X_train)
        self.accuracy = accuracy_score(y_train, y_pred_train)
        print('AdaBoost Spam Clasificator implementat cu biblioteci externe')
        print()
        print(f"Acuratețea la antrenare obținută este de: {self.accuracy * 100:.2f}%")

    def evaluate_on_test_data(self):
        X_test = self.vectorizer.transform(self.test_data['text'])
        y_test = self.test_data['is_spam'].astype('int')
        y_pred_test = self.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred_test)
        print(f"Acuratețea la testare obținută este de: {self.accuracy * 100:.2f}%")

    def cross_validate(self, cv=5):
        X, y = self.preprocess_data()
        scores = cross_val_score(self.classifier, X, y, cv=cv)
        mean_accuracy = scores.mean()
        print(f"Acuratețea la cross-validation obținută este de: {mean_accuracy * 100:.2f}%")

if __name__ == '__main__':
    n_estimators = 50
    ada_classifier = AdaBoostWithLibraries(n_estimators=n_estimators)
    ada_classifier.evaluate_on_train_data()
    ada_classifier.cross_validate()
    ada_classifier.evaluate_on_test_data()
