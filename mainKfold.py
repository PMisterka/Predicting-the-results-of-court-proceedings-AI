import csv
import os
import re
import string
import gensim
import spacy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import defaultdict
from gensim.utils import simple_preprocess
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, KFold

nlp = spacy.load('pl_core_news_sm')

stop_words_txt = open('stopwords.txt', 'r', encoding='utf8')
stoplist = stop_words_txt.read()

# Usunięcie interpunkcji, stokenizowanie słów
def tokenize_and_clean(text):
    cleaned_text = ' '.join([word for word in simple_preprocess(text) if word not in string.punctuation])
    return cleaned_text

def create_file_data(category_path, label):
    file_data = defaultdict(str)

    for filename in os.listdir(category_path):
        if filename.endswith('.txt'):
            with open(os.path.join(category_path, filename), 'r', encoding='utf-8-sig') as file:
                text = file.read()
                tokenized_text = tokenize_and_clean(text)
                file_data[filename] = {'text': tokenized_text, 'label': label}

    return file_data


file_data_k = create_file_data('kradziez', 'kradziez')
file_data_o = create_file_data('oszustwo', 'oszustwo')
file_data_z = create_file_data('zdrowie', 'przestępstwo przeciwko zdrowiu')
file_data_ko = create_file_data('komunikacja', 'przestępstwo w komunikacji')


#połączenie wszystkich słowników w jeden:
merged_dict = {**file_data_k, **file_data_o, **file_data_z, **file_data_ko}


tokenized_data = {key: {'text': (value['text']), 'label': value['label']} for key, value in merged_dict.items()}


# stworzenie nazwy pliku CSV
csv_filename = 'tokenized_data_final_sm.csv'

# wpisanie danch do pliku csv (wraz z podziałem na kolumny - nazwa, text oraz label)
with open(csv_filename, 'w', encoding='utf-8-sig', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # określenie kolumn
    writer.writerow(['Key', 'Text', 'Label'])

    # wpisanie wartości
    for key, values in merged_dict.items():
        doc = nlp(values['text'].lower())
        lem_text = [token.lemma_ for token in doc if token.lemma_ not in stoplist]
        writer.writerow([key,lem_text, values['label']])


# stworzenie korpusu
corpus = pd.read_csv("tokenized_data_final_sm.csv")
X = corpus.Text
y = corpus.Label

# całość Accuracy
correct_predictions_all = 0

y_true_all = []
y_pred_all = []

# określenie ilości foldów
kf = KFold(n_splits=10, shuffle=True)

print('Accuracy of every fold:')
for i, (train_index, test_index) in enumerate(kf.split(X)):
    print(f'Fold {i}:') #print numeru foldu (od 0 do 9)

# dzielimy korpus na treningowy i testowy
    Xtrain = X[train_index]
    Xtest = X[test_index]
    ytrain = y[train_index]
    ytest = y[test_index]

    texts = [[word for word in re.split('\W+',doc.lower())]
             for doc in Xtrain]

    textsTEST = [[word for word in re.split('\W+',doc.lower())]
             for doc in Xtest]

    # tworzymy nowy słownik - frequency
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    # usuwamy słowa które (w danym dokumencie) pojawiają się mniej niż x razy
    x = 5
    processed_corpus = [[token for token in text if frequency[token] > x]
                        for text in texts]

    # tworzymy słownik unikalnych tokenów
    dictionary = gensim.corpora.Dictionary(processed_corpus)

    # konwertujemy ten słownik do bag of words
    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]

    # tworzymy model
    model = gensim.models.LsiModel(bow_corpus)

    # tworzymy macierz podobieństw
    index = gensim.similarities.SparseMatrixSimilarity(model[bow_corpus],num_features=len(dictionary))

    # losujemy orzeczenie do przetestowania
    t = np.random.randint(len(Xtest))

    query_document = textsTEST[t]
    # konwerrtujemy orzeczenie testowe do BoW
    query_bow = dictionary.doc2bow(query_document)

    sims = index[model[query_bow]]
    docNumber = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[0][0]
    # print('Predicted:',y[docNumber])
    # print('Ground Truth:',ytest.iloc[t])

    # sprawdzanie accuracy
    correct_predictions = 0

    y_true = []
    y_pred = []

    for t in range(len(Xtest)):
        query_document = textsTEST[t]
        query_bow = dictionary.doc2bow(query_document)

        sims = index[model[query_bow]]

        docNumber = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[0][0]

        # jeśli model dobrze dopasował kategorie - dodajemy 1 do zmiennej correct_predictions (sprawdzanie accuracy danego foldu)
        if y[docNumber] == ytest.iloc[t]:
            correct_predictions += 1

        y_pred.append(y[docNumber])
        y_true.append(ytest.iloc[t])

        # jeśli model dobrze dopasował kategorie - dodajemy 1 do zmiennej correct_predictions_all aby sprawdzic calosciowa accuracy
        if y[docNumber] == ytest.iloc[t]:
            correct_predictions_all += 1

        y_pred_all.append(y[docNumber])
        y_true_all.append(ytest.iloc[t])

    # accuracy obliczona ręcznie dla danego foldu
    accuracy = correct_predictions / len(Xtest) * 100
    print(f'Accuracy: {accuracy:.2f}%')

# całościowa accuracy obliczona ręcznie
accuracy = correct_predictions_all / len(X) * 100
print(f'\nOverall Accuracy: {accuracy:.2f}%')

# print(multilabel_confusion_matrix(y_true_all, y_pred_all, labels=["kradziez", "oszustwo", "przestępstwo przeciwko zdrowiu", "przestępstwo w komunikacji" ]))

target_names = labels=["kradziez", "oszustwo", "p. przeciwko zdrowiu", "p. w komunikacji" ]

# confusion matrix wyświetlająca liczbę poprawnie i niepoprawnie sklasyfikowanych orzeczeń.
print('\nConfusion matrix with sklearn:')
print(confusion_matrix(y_true_all, y_pred_all))

# classification report wyświetlający precyzje, czułość, f1 score, accuracy
print('\nClassification report with sklearn:')
print(classification_report(y_true_all, y_pred_all))

# znormalizowana confusion matrix (seaborn)
cm = confusion_matrix(y_true_all, y_pred_all)
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(7,5))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()