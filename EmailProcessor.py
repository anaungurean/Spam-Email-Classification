import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation


def remove_numbers(tokens):
    return [word for word in tokens if word.isalpha()]


class EmailProcessor:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.data_processed = self.load_data()
        self.save_to_csv(self.data_processed, 'test_data_processed.csv')

    def load_data(self):
        data = []
        root_folder_path = self.folder_path
        for root, dirs, files in os.walk(root_folder_path):
            print(root)
            for file in files:
                file_path = os.path.join(root, file)
                print(file_path)
                with open(file_path, 'r', encoding='utf-8') as f:
                  print(file_path)
                  if 'part10' in file_path:
                    content = f.read()
                    is_spam = 'spm' in file
                    processed_text = self.process_text(content)
                    data.append([processed_text, is_spam])
                    print(f"Processed {file_path}")
                    print("=" * 50)
                    print(f"Processed text: {processed_text}")
                    print("=" * 50)
                    print(f"Is spam: {is_spam}")
        df = pd.DataFrame(data, columns=['text', 'is_spam'])
        return df

    def process_text(self, text):
        text = self.to_lowercase(text)
        text = self.remove_punctuation(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = remove_numbers(tokens)
        return ' '.join(tokens)

    def to_lowercase(self, text):
        return text.lower()

    def remove_punctuation(self, text):
        return ''.join([char for char in text if char not in punctuation])

    def tokenize(self, text):
        return word_tokenize(text)

    def remove_stopwords(self, tokens):
        return [word for word in tokens if word not in stopwords.words('english')]

    def save_to_csv(self, df, file_name):
        df.to_csv(file_name, index=False)
        print(f"Saved to {file_name}")

if __name__ == '__main__':
    folder_path = "lingspam_public"
    processor = EmailProcessor(folder_path)

