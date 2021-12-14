# import click
# import pandas as pd
#
# from models.no_decomposition_model import ModelNoDecomposition
#
# @click.command()
# @click.option('--file', required=True)
# @click.option('--desc/--asc', default=False, required=True)
# def run_task(file, desc):
#     df = pd.read_csv(file, encoding='utf-8', low_memory=False)
#
#     if desc:
#         df = df.sort_values(by=['LD50'], ascending=False)
#     else:
#         df = df.sort_values(by=['LD50'], ascending=True)
#
#     model = ModelNoDecomposition(df)
#     r = model.train()
#     for idx, item in enumerate(r):
#         click.echo(f'fold_{idx+1} : {item}')
#
# if __name__ == '__main__':
#     run_task()

# import csv
#
# from ast import literal_eval
#
# import pandas as pd
#
# with open('apis_contact.csv', newline='') as c:
#     rows = csv.DictReader(c)
#
#     big_list = []
#     for row in rows:
#
#         temp_dict = {}
#         for k,v in row.items():
#             temp_dict.update({k:literal_eval(v)})
#
#         big_list.append(temp_dict)
#
#     df = pd.DataFrame(big_list)
#     print(df.head())

import pandas as pd

# df = pd.read_csv('apis_contact.csv', encoding='utf-8', low_memory=False)
# df.to_json('apis_contact.json', indent=4)
df = pd.read_json('apis_contact.json', encoding='utf-8')
print(df.iloc[])
