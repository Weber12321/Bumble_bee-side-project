import click
import pandas as pd

from models.no_decomposition_model import ModelNoDecomposition

@click.command()
@click.option('--file', required=True)
def run_task(file):
    df = pd.read_csv(file, encoding='utf-8', low_memory=False)
    df = df.dropna()
    model = ModelNoDecomposition(df)
    click.echo(f'The results of 5-fold are {model.train()}')



if __name__ == '__main__':
    run_task()