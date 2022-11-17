
import pdb
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.naive_bayes import BernoulliNB
from nltk.tokenize.sonority_sequencing import SyllableTokenizer
from sklearn.tree import DecisionTreeClassifier
from functools import reduce
from operator import add

train_data = pd.read_csv("data/train.csv")
val_and_test_data = pd.read_csv("data/val_and_test_input.csv")
with open("data/category2_encoding.json") as keyfile:
    key = json.load(keyfile)
inverse_key = {v:k for k,v in key.items()}

documents = train_data.product_title

def tokenize(text):
    return reduce(add, [t._.syllables if t._.syllables else [t.text] for t in nlp(text)])

vec = TfidfVectorizer(
        #preprocessor = preprocessor,
        tokenizer = SyllableTokenizer().tokenize,
        min_df=.05)
        

clf = DecisionTreeClassifier()

train_X = vec.fit_transform(documents)
train_y = [key[v] for v in train_data.category2]

pdb.set_trace()

clf.fit(train_X,train_y)
val = vec.transform(val_and_test_data.product_title)
val_y = clf.predict(val)

val_and_test_data["predicted"] = [inverse_key[v] for v in clf.predict(val)]
val_and_test_data.to_csv("my-first-predictions.csv")
