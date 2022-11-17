import click
import json
import joblib

@click.command()
@click.argument("syllables", type = click.File("r"))
@click.option("-m","--model",type=click.File("rb"), default = "model.jbl")
@click.option("-o","--out",type=click.File("w"), default="-")
def main(syllables, model, out):
    data = json.load(syllables)
    model = joblib.load(model)

    X = [d["syllables"] for d in data]

    predictions = [int(p) for p in model.predict(X)]

    data = [{**d, **{"predicted":p}} for d,p in zip(data, predictions)]
    json.dump(data, out)

if __name__ == "__main__":
    main()
