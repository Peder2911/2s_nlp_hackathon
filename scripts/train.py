
import json
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
import click
from toolz.functoolz import identity

@click.command()
@click.argument("syllables", type = click.File("r"))
@click.option("-o","--out", type = click.File("wb"), required = True, default = "model.jbl")
def main(syllables, out):
    syllable_data = json.load(syllables)

    pipeline = Pipeline([
            ("vectorizer", CountVectorizer(preprocessor=identity, tokenizer=identity)),
            ("tree", DecisionTreeClassifier())])

    X = [d["syllables"] for d in syllable_data]
    y = [d["label"] for d in syllable_data]

    pipeline.fit(X,y)
    joblib.dump(pipeline, out)

if __name__ == "__main__":
    main()
