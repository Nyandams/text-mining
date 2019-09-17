import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

df = pd.read_csv('data/mormon.csv', delimiter=";", skiprows=1, names=['Text', 'Author'], encoding='latin-1')

X = df['Text'].astype('U')
y = df['Author']

lemmatiser = WordNetLemmatizer()  # Defining a module for Text Processing


def text_process(tex):
    # 1. Removal of Punctuation Marks
    nopunct = [char for char in tex if char not in string.punctuation]
    nopunct = ''.join(nopunct)
    # 2. Lemmatisation
    a = ''
    i = 0
    for i in range(len(nopunct.split())):
        b = lemmatiser.lemmatize(nopunct.split()[i], pos="v")
        a = a + b + ' '
    # 3. Removal of Stopwords
    return [word for word in a.split() if word.lower() not
            in stopwords.words('english')]


labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

# 80-20 splitting the dataset (80%->Training and 20%->Validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# defining the bag-of-words transformer on the text-processed corpus # i.e., text_process() declared in II is executed..
bow_transformer = CountVectorizer(analyzer=text_process).fit(X_train)

# transforming into Bag-of-Words and hence textual data to numeric..
text_bow_train = bow_transformer.transform(X_train)  # ONLY TRAINING DATA# transforming into Bag-of-Words
# and hence textual data to numeric..
text_bow_test = bow_transformer.transform(X_test)  # TEST DATA

# instantiating the model with Multinomial Naive Bayes..
model = MultinomialNB()

# training the model...
model = model.fit(text_bow_train, y_train)

print('training accuracy'.format(model.score(text_bow_train, y_train)))  # Training Accuracy
print('training accuracy'.format(model.score(text_bow_test, y_test)))   # Validation Accuracy


# getting the predictions of the Validation Set...
predictions = model.predict(text_bow_test)
# getting the Precision, Recall, F1-Score
print(classification_report(y_test,predictions))