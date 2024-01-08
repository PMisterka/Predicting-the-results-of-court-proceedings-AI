import csv
import os
import string
import spacy
from collections import defaultdict
from gensim.utils import simple_preprocess

nlp = spacy.load('pl_core_news_sm')

stop_words_txt = open('stopwords.txt', 'r', encoding='utf8')
stoplist = stop_words_txt.read()

# stworzenie nazwy pliku CSV
csv_filename = 'GPT.csv'

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


file_data_k = create_file_data('GPT/kradziez', 'kradziez')
file_data_o = create_file_data('GPT/oszustwo', 'oszustwo')
file_data_z = create_file_data('GPT/zdrowie', 'przestępstwo przeciwko zdrowiu')
file_data_ko = create_file_data('GPT/komunikacja', 'przestępstwo w komunikacji')


#połączenie wszystkich słowników w jeden:
merged_dict = {**file_data_k, **file_data_o, **file_data_z, **file_data_ko}


tokenized_data = {key: {'text': (value['text']), 'label': value['label']} for key, value in merged_dict.items()}

# wpisanie danch do pliku csv (wraz z podziałem na kolumny - nazwa, text oraz label)
with open(csv_filename, 'w', encoding='utf-8-sig', newline='') as csvfile: #
    writer = csv.writer(csvfile)

    # określenie kolumn
    writer.writerow(['Key', 'Text', 'Half', 'Quarter', 'Label'])

    # wpisanie wartości
    for key, values in merged_dict.items():
        doc = nlp(values['text'].lower())
        # cały tekst
        lem_text = [token.lemma_ for token in doc if token.lemma_ not in stoplist]
        # 50% tekstu
        half = (0.5 * len(lem_text))
        half_text = [word for word in lem_text[:int(half)]]
        # 25% tekstu
        quarter = (0.25 * len(lem_text))
        quarter_text = [word for word in lem_text[:int(quarter)]]
        # 10% tekstu
        ten = (0.1 * len(lem_text))
        ten_text = [word for word in lem_text[:int(ten)]]

        writer.writerow([key, lem_text, half_text, quarter_text, ten_text, values['label']])


with open(csv_filename, 'r', encoding='utf-8-sig') as csvfile:
    csv_reader = csv.reader(csvfile)

    next(csv_reader)

    # print wybranej kolumny z określoną długością orzeczeń
    for row in csv_reader:
        labels = (row[5]) #odczytanie kolumny z label
        text_token = eval(row[4]) #odczytanie kolumny z tekstem
        text_str = ' '.join(text_token)
        print("LABEL: ", labels, "\nTEXT: ", text_str)


