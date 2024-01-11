import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from joblib import dump

# Citirea datelor
data = pd.read_csv('train_data_processed.csv')

# Crearea unui vectorizer
vectorizer = CountVectorizer(stop_words='english')

# Transformarea textului într-o reprezentare numerică
X_vectorized = vectorizer.fit_transform(data['text'])

# Antrenarea clasificatorului Naive Bayes
classifier = MultinomialNB()
classifier.fit(X_vectorized, data['is_spam'])

# Salvarea modelului antrenat, dacă dorești
dump(classifier, 'naive_bayes_model.joblib')

# Realizarea predicțiilor pe datele de antrenare
predictions = classifier.predict(X_vectorized)

# Calcularea și afișarea acurateței la antrenare
accuracy = metrics.accuracy_score(data['is_spam'], predictions)
print(f'Acuratețe la antrenare: {accuracy * 100:.2f}%')
