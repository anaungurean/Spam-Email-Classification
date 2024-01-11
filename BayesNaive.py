import csv
from collections import defaultdict
import matplotlib.pyplot as plt

class NaiveBayesSpamClassifier:
    def __init__(self, file_name='train_data_processed.csv'):
        self.file_name = file_name
        self.total_spam = 0
        self.total_non_spam = 0
        self.spam_word_counts = defaultdict(int)
        self.non_spam_word_counts = defaultdict(int)
        self.data = self.load_data()
        self.error_rates = []

    def load_data(self):
        data = []
        with open(self.file_name, 'r') as f:
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

    def predict(self, text):
        words = text.split()

        spam_probability = self.total_spam / (self.total_spam + self.total_non_spam)
        non_spam_probability = self.total_non_spam / (self.total_spam + self.total_non_spam)

        for word in words:
            spam_probability *= self.spam_word_counts[word] / self.total_spam
            non_spam_probability *= self.non_spam_word_counts[word] / self.total_non_spam
        return spam_probability > non_spam_probability

    def calculate_error(self, test_data):
        errors = 0
        for row in test_data:
            text = row['text']
            is_spam = row['is_spam'] == 'True'
            prediction = self.predict(text)
            if prediction != is_spam:
                errors += 1

        error_rate = errors / len(test_data)
        return error_rate

    def cross_validate(self):
        num_iterations = len(self.data)

        for i in range(num_iterations):
            print(f"Cross-validation iteratie {i+1}/{num_iterations}")
            train_data = self.data[:i] + self.data[i+1:]
            test_data = [self.data[i]]
            self.total_spam = 0
            self.total_non_spam = 0
            self.spam_word_counts = defaultdict(int)
            self.non_spam_word_counts = defaultdict(int)

            self.train(train_data)
            error_rate = self.calculate_error(test_data)
            self.error_rates.append(error_rate)

        average_error_rate = sum(self.error_rates) / num_iterations
        print(f"Acuratete medie folosind Leave-One-Out cross-validation: {100 - average_error_rate * 100:.2f}%")

        self.plot_cross_validation_results()

    def plot_cross_validation_results(self):
        plt.plot(range(1, len(self.error_rates) + 1), self.error_rates, marker='o')
        plt.title('Leave-One-Out Cross-Validation')
        plt.xlabel('Iteratii')
        plt.ylabel('Acuratete')
        plt.show()

if __name__ == '__main__':
    classifier = NaiveBayesSpamClassifier()
    classifier.cross_validate()
