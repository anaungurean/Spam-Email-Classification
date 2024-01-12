import csv
from collections import defaultdict
import matplotlib.pyplot as plt

class NaiveBayesWithoutLibraries:
    def __init__(self, train_file_name='new_train_data_processed.csv', test_file_name='new_test_data_processed.csv'):
        self.train_file_name = train_file_name
        self.test_file_name = test_file_name
        self.total_spam = 0
        self.total_non_spam = 0
        self.spam_word_counts = defaultdict(int)
        self.non_spam_word_counts = defaultdict(int)
        self.train_data = self.load_data(self.train_file_name)
        self.test_data = self.load_data(self.test_file_name)
        self.error_rates = []

    def load_data(self, file_name):
        data = []
        with open(file_name, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                text = row[0]
                is_spam = row[1]
                data.append({'text': text, 'is_spam': is_spam})
        return data

    def train(self, data):
        for row in data:
            words = row['text'].split()
            is_spam = row['is_spam']
            if is_spam == 'True':
                self.total_spam += 1
                for word in words:
                    self.spam_word_counts[word] += 1
            else:
                self.total_non_spam += 1
                for word in words:
                    self.non_spam_word_counts[word] += 1

    def predict_with_laplace(self, text):
        words = text.split()

        spam_probability = self.total_spam / (self.total_spam + self.total_non_spam)
        non_spam_probability = self.total_non_spam / (self.total_spam + self.total_non_spam)

        for word in words:
            spam_probability *= (self.spam_word_counts[word] + 1) / (self.total_spam + len(self.spam_word_counts))
            non_spam_probability *= (self.non_spam_word_counts[word] + 1) / (self.total_non_spam + len(self.non_spam_word_counts))

        return spam_probability > non_spam_probability

    def predict_without_laplace(self, text):
        words = text.split()

        spam_probability = self.total_spam / (self.total_spam + self.total_non_spam)
        non_spam_probability = self.total_non_spam / (self.total_spam + self.total_non_spam)

        for word in words:
            spam_probability *= self.spam_word_counts[word] / self.total_spam
            non_spam_probability *= self.non_spam_word_counts[word]  / self.total_non_spam

        return spam_probability > non_spam_probability

    def calculate_error_with_Laplace(self, test_data):
        errors = 0
        for row in test_data:
            text = row['text']
            is_spam = row['is_spam'] == 'True'
            prediction = self.predict_with_laplace(text)
            if prediction != is_spam:
                errors += 1

        error_rate = errors / len(test_data)
        return error_rate

    def calculate_error_without_Laplace(self, test_data):
        errors = 0
        for row in test_data:
            text = row['text']
            is_spam = row['is_spam'] == 'True'
            prediction = self.predict_without_laplace(text)
            if prediction != is_spam:
                errors += 1

        error_rate = errors / len(test_data)
        return error_rate


    def cross_validate(self):
        num_iterations = len(self.train_data)

        for i in range(num_iterations):
            print(f"Cross-validation iteration {i+1}/{num_iterations}")
            train_data = self.train_data[:i] + self.train_data[i+1:]
            test_data = [self.train_data[i]]

            self.total_spam = 0
            self.total_non_spam = 0
            self.spam_word_counts = defaultdict(int)
            self.non_spam_word_counts = defaultdict(int)

            self.train(train_data)
            error_rate = self.calculate_error_with_Laplace(test_data)
            self.error_rates.append(error_rate)

        average_error_rate = sum(self.error_rates) / num_iterations
        print(f"Acuratețea la cross-validation obținută este de: {100 - average_error_rate * 100:.2f}%")

        self.plot_cross_validation_results()

    def plot_cross_validation_results(self):
        plt.plot(range(1, len(self.error_rates) + 1), self.error_rates, marker='o')
        plt.title('Leave-One-Out Cross-Validation')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.show()

    def evaluate_on_test_data(self):
        self.total_spam = 0
        self.total_non_spam = 0
        self.spam_word_counts = defaultdict(int)
        self.non_spam_word_counts = defaultdict(int)

        self.train(self.train_data)
        test_error_rate = self.calculate_error_with_Laplace(self.test_data)
        print(f"Acuratețea la testare obținută este de: {100 - test_error_rate * 100:.2f}%")

    def evaluate_on_train_data(self):
        self.total_spam = 0
        self.total_non_spam = 0
        self.spam_word_counts = defaultdict(int)
        self.non_spam_word_counts = defaultdict(int)
        self.train(self.train_data)
        train_error_rate = self.calculate_error_without_Laplace(self.train_data)
        print('Bayes Naiv Spam Clasificator implementat fără biblioteci externe')
        print()
        print(f"Acuratețea la antrenare obținută este de: {100 - train_error_rate * 100:.2f}%")

    def cross_validate_optimal(self):
        self.total_spam = 0
        self.total_non_spam = 0
        self.spam_word_counts = defaultdict(int)
        self.non_spam_word_counts = defaultdict(int)

        self.train(self.train_data)
        num_iterations = len(self.train_data)
        initial_spam_word_counts = self.spam_word_counts.copy()
        initial_non_spam_word_counts = self.non_spam_word_counts.copy()
        initial_total_spam = self.total_spam
        initial_total_non_spam = self.total_non_spam

        for i in range(num_iterations):
            test_data = [self.train_data[i]]
            for row in test_data:
                words = row['text'].split()
                is_spam = row['is_spam']
                if is_spam == 'True':
                    self.total_spam -= 1
                else:
                    self.total_non_spam -= 1
                for word in words:
                    if is_spam == 'True':
                        self.spam_word_counts[word] -= 1
                    else:
                        self.non_spam_word_counts[word] -= 1

            error_rate = self.calculate_error_without_Laplace(test_data)
            self.error_rates.append(error_rate)
            self.spam_word_counts = initial_spam_word_counts.copy()
            self.non_spam_word_counts = initial_non_spam_word_counts.copy()
            self.total_spam = initial_total_spam
            self.total_non_spam = initial_total_non_spam

        average_error_rate = sum(self.error_rates) / num_iterations
        print(f"Acuratețea la cross-validation obținută este de: {100 - average_error_rate * 100:.2f}%")


if __name__ == '__main__':
    classifier = NaiveBayesWithoutLibraries()
    classifier.evaluate_on_train_data()
    classifier.cross_validate_optimal()
    # classifier.cross_validate()
    classifier.evaluate_on_test_data()
