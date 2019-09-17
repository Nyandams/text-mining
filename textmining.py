import pandas as pd
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.collocations import *
from sklearn.feature_extraction.text import TfidfVectorizer
try:
    nltk.download('wordnet')
except LookupError:
    import nltk
    nltk.download('wordnet')

try:
    stopwords = set(stopwords.words('english'))
except LookupError:
    import nltk
    nltk.download('stopwords')
    stopwords = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

train = pd.read_csv('data/mormon.csv', delimiter=";", skiprows=1, names=['Text', 'Author'],
                    encoding='latin-1')

# print('Author of sample:', train['Author'][0])
# print(train)

#print(train.head())

""" Wordcloud
all_text = ' '.join([str(text) for text in train['Text']])
print('Number of words in all_text:', len(all_text))

wordcloud = WordCloud(width=800, height=500,
                      random_state=21, max_font_size=110).generate(all_text)
plt.figure(figsize=(15, 12))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
"""


""" word frequency
mormon = train[train['Author'] == 'Mormon']
mormon_text = ' '.join(str(text) for text in mormon['Text'])

mormon_word_list = word_tokenize(mormon_text)
mormon_word_list = [x.lower() for x in mormon_word_list]

mormon_clean = [w.lower() for w in mormon_word_list if w not in stopwords and w.isalpha()]
mormon_clean = ' '.join(text.lower() for text in mormon_clean)

mormon_list = mormon_clean.split()
mormon_counts = Counter(mormon_list)
mormon_common_words = [word[0] for word in mormon_counts.most_common(25)]
mormon_common_counts = [word[1] for word in mormon_counts.most_common(25)]

plt.style.use('dark_background')
plt.figure(figsize=(15, 12))

sns.barplot(x=mormon_common_words, y=mormon_common_counts)
plt.title('Most Common Words used by Mormon')

print(stopwords)
plt.show()
"""

lemm = WordNetLemmatizer()


class LemmaCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(LemmaCountVectorizer, self).build_analyzer()
        return lambda doc: (lemm.lemmatize(w) for w in analyzer(doc))


text = list(train.Text.values.astype('U'))

# Calling our overwritten Count vectorizer
tf_vectorizer = LemmaCountVectorizer(max_df=0.95,
                                     min_df=2,
                                     stop_words='english',
                                     decode_error='ignore')
tf = tf_vectorizer.fit_transform(text)

lda = LatentDirichletAllocation(n_components=9, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)

lda.fit(tf)
"""
LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
                          evaluate_every=-1, learning_decay=0.7,
                          learning_method='online', learning_offset=50.0,
                          max_doc_update_iter=100, max_iter=5,
                          mean_change_tol=0.001, n_components=9, n_jobs=None,
                          perp_tol=0.1, random_state=0, topic_word_prior=None,
                          total_samples=1000000.0, verbose=0)
"""


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


n_top_words = 40
print("\nTopics in LDA model: ")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)

first_topic = lda.components_[0]
