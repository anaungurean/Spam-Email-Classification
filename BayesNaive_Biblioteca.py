import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score


class NaiveBayesClassifier:
    def __init__(self, file_name='train_data_processed.csv'):
        self.file_name = file_name
        self.data = self.load_data()
        self.vectorizer = CountVectorizer()
        self.classifier = MultinomialNB()
        self.accuracy = None

    def load_data(self):
        data = pd.read_csv(self.file_name)
        return data

    def preprocess_data(self):
        X = self.vectorizer.fit_transform(self.data['text'])
        y = self.data['is_spam'].astype('int')
        return X, y

    def train(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)

    def predict(self, X_test):
        return self.classifier.predict(X_test)

    def calculate_accuracy_train(self, y_true, y_pred):
        self.accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy at training: {self.accuracy * 100:.2f}%")

    def train_and_evaluate(self):
        X, y = self.preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.train(X_train, y_train)
        y_pred_train = self.predict(X_train)
        self.calculate_accuracy_train(y_train, y_pred_train)

    def cross_validate(self, cv=5):
        X, y = self.preprocess_data()
        scores = cross_val_score(self.classifier, X, y, cv=cv)
        mean_accuracy = scores.mean()
        print(f"Cross-validated Accuracy: {mean_accuracy * 100:.2f}%")

if __name__ == '__main__':
    classifier = NaiveBayesClassifier()
    classifier.train_and_evaluate()
    classifier.cross_validate()