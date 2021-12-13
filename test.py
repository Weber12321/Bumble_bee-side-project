import click
import pandas as pd

from models.no_decomposition_model import ModelNoDecomposition

@click.command()
@click.option('--file', required=True)
@click.option('--desc/--asc', default=False, required=True)
def run_task(file, desc):
    df = pd.read_csv(file, encoding='utf-8', low_memory=False)

    if desc:
        df = df.sort_values(by=['LD50'], ascending=False)
    else:
        df = df.sort_values(by=['LD50'], ascending=True)

    model = ModelNoDecomposition(df)
    r = model.train()
    for idx, item in enumerate(r):
        click.echo(f'fold_{idx+1} : {item}')



if __name__ == '__main__':
    run_task()