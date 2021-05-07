from collections import Counter
import nltk
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from prettytable import PrettyTable
from matplotlib import rcParams
import matplotlib.cm as cm
import string
import re
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn import preprocessing
import pandas
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv('phrase.csv')


carrier_count = data["class"].value_counts()
sns.set(style="darkgrid")
sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)
plt.title('Frequency Distribution of class')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Carrier', fontsize=12)
plt.show()

data["class"].value_counts().head(3).plot(
    kind='pie', autopct='%1.1f%%', figsize=(8, 8)).legend()

data["class"].value_counts()

nltk.download('punkt')

nltk.word_tokenize(data["phrase"][0])

data["phrase"][0]

# Tokenize using the white spaces
nltk.tokenize.WhitespaceTokenizer().tokenize(data["phrase"][0])

# Tokenize using Punctuations
nltk.tokenize.WordPunctTokenizer().tokenize(data["phrase"][0])


# Tokenization using grammer rules
nltk.tokenize.TreebankWordTokenizer().tokenize(data["phrase"][0])

words = nltk.tokenize.WhitespaceTokenizer().tokenize(data["phrase"][0])
df = pd.DataFrame()
df['OriginalWords'] = pd.Series(words)
# porter's stemmer
porterStemmedWords = [nltk.stem.PorterStemmer().stem(word) for word in words]
df['PorterStemmedWords'] = pd.Series(porterStemmedWords)
# SnowBall stemmer
snowballStemmedWords = [nltk.stem.SnowballStemmer(
    "arabic").stem(word) for word in words]
df['SnowballStemmedWords'] = pd.Series(snowballStemmedWords)
df

nltk.download('wordnet')

# LEMMATIZATION
words = nltk.tokenize.WhitespaceTokenizer().tokenize(data["phrase"][0])
df = pd.DataFrame()
df['OriginalWords'] = pd.Series(words)
# WordNet Lemmatization
wordNetLemmatizedWords = [
    nltk.stem.WordNetLemmatizer().lemmatize(word) for word in words]
df['WordNetLemmatizer'] = pd.Series(wordNetLemmatizedWords)
df

print(len(data))

data['phrase'].head()

data[data.isnull().any(axis=1)].head()

np.sum(data.isnull().any(axis=1))

for letter in '#.][!XR':
    data['phrase'] = data['phrase'].astype(str).str.replace(letter, '')

data.head()

arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations


def remove_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)


def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text


def remove_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)


def processPost(tweet):

    # Replace @username with empty string
    tweet = re.sub('@[^\s]+', ' ', tweet)

    # Convert www.* or https?://* to " "
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', tweet)

    # Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

    # remove punctuations
    tweet = remove_punctuations(tweet)

    # normalize the tweet
    #tweet= normalize_arabic(tweet)

    # remove repeated letters
    tweet = remove_repeating_char(tweet)

    return tweet


# apply used to call the method processpost
data["phrase"] = data["phrase"].apply(lambda x: processPost(x))

tokenizer = RegexpTokenizer(r'\w+')
data["phrase"] = data["phrase"].apply(tokenizer.tokenize)

data["phrase"].head()

nltk.download('stopwords')

stopwords_list = stopwords.words('arabic')

stopwords_list

print(len(stopwords_list))

print(type(stopwords_list))

listToStr = ' '.join([str(elem) for elem in stopwords_list])

listToStr

data['phrase'] = data['phrase'].apply(
    lambda x: [item for item in x if item not in stopwords_list])

all_words = [word for tokens in data['phrase'] for word in tokens]
sentence_lengths = [len(tokens) for tokens in data['phrase']]

VOCAB = sorted(list(set(all_words)))

print("%s words total, with a vocabulary size of %s" %
      (len(all_words), len(VOCAB)))
print("Max sentence length is %s" % max(sentence_lengths))

counter = Counter(all_words)

counter.most_common(25)

counted_words = Counter(all_words)

words = []
counts = []
for letter, count in counted_words.most_common(25):
    words.append(letter)
    counts.append(count)

colors = cm.rainbow(np.linspace(0, 1, 10))
rcParams['figure.figsize'] = 20, 10

plt.title('Top words in positive')
plt.xlabel('Count')
plt.ylabel('Words')
plt.barh(words, counts, color=colors)

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    ngram_range=(1, 1),
    max_features=10000)

unigramdataGet = word_vectorizer.fit_transform(data['phrase'].astype('str'))
unigramdataGet = unigramdataGet.toarray()

vocab = word_vectorizer.get_feature_names()
unigramdata_features = pd.DataFrame(np.round(unigramdataGet, 1), columns=vocab)
unigramdata_features[unigramdata_features > 0] = 1

unigramdata_features.head()

pro = preprocessing.LabelEncoder()
data['class'].replace({-1: 2}, inplace=True)
encpro = pro.fit_transform(data['class'])
data['class'] = encpro


y = data['class']
X = unigramdata_features
y

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=2)

nb = GaussianNB()
nb = nb.fit(X_train, y_train)
nb

y_pred = nb.predict(X_test)
nb_1 = nb.score(X_test, y_test)
print('Accuracy= {:.3f}'.format(nb.score(X_test, y_test)))
print("Precision Score : ", precision_score(y_test, y_pred, average=None))
print("recall_score : ", recall_score(y_test, y_pred, average=None))
print("F1 : ", f1_score(y_test, y_pred, average=None))

RC = RidgeClassifier()
RC = RC.fit(X_train, y_train)
RC

y_pred = RC.predict(X_test)
rc_1 = RC.score(X_test, y_test)
print('Accuracy= {:.3f}'.format(RC.score(X_test, y_test)))
print("Precision Score : ", precision_score(y_test, y_pred, average=None))
print("recall_score : ", recall_score(y_test, y_pred, average=None))
print("F1 : ", f1_score(y_test, y_pred, average=None))

PC = PassiveAggressiveClassifier()
PC = PC.fit(X_train, y_train)
PC

y_pred = PC.predict(X_test)
pc_1 = PC.score(X_test, y_test)
print('Accuracy= {:.3f}'.format(PC.score(X_test, y_test)))

print("Precision Score : ", precision_score(y_test, y_pred, average=None))
print("recall_score : ", recall_score(y_test, y_pred, average=None))
print("F1 : ", f1_score(y_test, y_pred, average=None))

LR = LogisticRegression(penalty='l2', C=1)
LR = LR.fit(X_train, y_train)
LR

y_pred = LR.predict(X_test)
lr_1 = LR.score(X_test, y_test)
print('Accuracy= {:.3f}'.format(LR.score(X_test, y_test)))

print("Precision Score : ", precision_score(y_test, y_pred, average=None))
print("recall_score : ", recall_score(y_test, y_pred, average=None))
print("F1 : ", f1_score(y_test, y_pred, average=None))
CR = classification_report(y_test, y_pred)
print(CR)
print('\n')
confusion_matrix(y_test, y_pred)
print(confusion_matrix)


svc = LinearSVC(C=1, max_iter=500)
svc = svc.fit(X_train, y_train)
svc

y_pred = svc.predict(X_test)
svc_1 = svc.score(X_test, y_test)
print('Accuracy= {:.3f}'.format(svc.score(X_test, y_test)))

print("Precision Score : ", precision_score(y_test, y_pred, average=None))
print("recall_score : ", recall_score(y_test, y_pred, average=None))
print("F1 : ", f1_score(y_test, y_pred, average=None))
confusion_matrix(y_test, y_pred)
print(confusion_matrix)


raf = RandomForestClassifier(
    min_samples_leaf=20, min_samples_split=20, random_state=10)
raf = raf.fit(X_train, y_train)
raf

y_pred = raf.predict(X_test)
raf_1 = raf.score(X_test, y_test)
print('Accuracy= {:.3f}'.format(raf.score(X_test, y_test)))

print("Precision Score : ", precision_score(y_test, y_pred, average=None))
print("recall_score : ", recall_score(y_test, y_pred, average=None))
print("F1 : ", f1_score(y_test, y_pred, average=None))


Ens = VotingClassifier(estimators=[(
    'SVM', svc), ('nb', nb), ('RC', RC), ('raf', raf), ('PC', PC), ('LR', LR)], voting='hard')
Ens = Ens.fit(X_train, y_train)
Ens

y_pred = Ens.predict(X_test)
Ens_1 = Ens.score(X_test, y_test)
print('Accuracy= {:.3f}'.format(Ens.score(X_test, y_test)))

print("Precision Score : ", precision_score(y_test, y_pred, average=None))
print("recall_score : ", recall_score(y_test, y_pred, average=None))
print("F1 : ", f1_score(y_test, y_pred, average=None))

x = PrettyTable()
print('\n')
print("Comparison of all algorithms on F1 score")
x.field_names = ["Model", "Accuracy"]


x.add_row(["Naive Bayes Algorithm", round(nb_1, 2)])
x.add_row(["Ridge Classifier Algorithm",  round(rc_1, 2)])
x.add_row(["Passive Aggressive Classifier Algorithm", round(pc_1, 2)])
x.add_row(["Logistic Regression Algorithm", round(lr_1, 2)])
x.add_row(["SVM", round(svc_1, 2)])
x.add_row(["Random Forest", round(raf_1, 2)])
x.add_row(["Ensemble d'algorithmes", round(Ens_1, 2)])

print(x)
print('\n')

PC = PassiveAggressiveClassifier()
PC = PC.fit(X_train, y_train)


testing = [
    'ضربُ النِّساء فعلٌ قبيح، وهو ساقطٌ من أعداد الرجال بل هو أقرب للحيوان منهُ']

x = word_vectorizer.transform(testing)

pred = PC.predict(x)
pred = pro.inverse_transform(pred)
prediction = pd.DataFrame(pred, columns=['class'])
print(prediction)

result = pd.DataFrame()
result['Text'] = testing
result['Prediction'] = prediction
result
