from typing import Optional
import re
import string
import json
import csv
import click
from nltk.tokenize.sonority_sequencing import SyllableTokenizer
from multiprocessing import Pool
from functools import partial, reduce
from operator import add

tkn = SyllableTokenizer()

tkn.vowels += "æøå"

for number in string.digits:
    tkn.phoneme_map[number] = 1

number_set = set(string.digits)
metrics = {
        "g","l","kg","dl","gram","liter","kilo", "pk", "m", "cm"
    }

def syllable_tokenize(id_idx, text_idx, label_idx, row):
    text = row[text_idx].lower()
    #text = re.sub(r"([0-9])([a-z])",r"\g<1> \g<2>", text)
    text = re.sub(r"[0-9\.]+",r" NUM ", text)
    words = text.split()
    syllables = reduce(add, [tkn.tokenize(word) for word in words])

    return row[id_idx], syllables, row[label_idx] if label_idx else None

@click.command()
@click.argument("dataset", type = click.File("r"))
@click.option("-k","--key", type = click.File("r"))
@click.option("-i","--id-column", type=str, default="id")
@click.option("-t","--text-column", type=str,required=True)
@click.option("-l","--label-column",type=str)
@click.option("-n","--n-cores",type=int, default=8)
@click.option("-o","--out",type=click.File("w"), default="-")
def main(dataset, key, id_column: str, text_column: str, label_column: Optional[str], n_cores:int, out):
    p = Pool(n_cores)
    data = csv.reader(dataset)
    key = json.load(key) if key else None
    header = next(data)

    id_idx = header.index(id_column)
    text_idx = header.index(text_column)
    label_idx = header.index(label_column) if label_column else None

    result = p.map(partial(syllable_tokenize, id_idx, text_idx, label_idx), data)
    click.echo(json.dump([{"id": i, "syllables": s, "label": key[l] if l and key else l} for i,s,l in result], out))

if __name__ == "__main__":
    main()
