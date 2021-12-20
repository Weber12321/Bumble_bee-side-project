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

# import pandas as pd

# df = pd.read_csv('apis_contact.csv', encoding='utf-8', low_memory=False)
# df.to_json('apis_contact.json', indent=4)
# from models.model_generator import generator
#
# df = pd.read_json('apis_contact.json', encoding='utf-8')
#
# model = generator().create_model('decomposition', df)
#
# header = list(df.columns)
# x_header = header[1:-1]
# scores = model.top_k.scores_
#
# feature_weight_dict = {}
# if len(x_header) == len(model.top_k.scores_):
#     for idx, element in enumerate(model.top_k.scores_):
#         feature_weight_dict.update({
#             x_header[idx] : element
#         })
# else:
#     raise ValueError(f'length of {x_header} is not equal to len of {model.top_k.scores_}')
#
# sort_dict = {k: v for k, v in sorted(feature_weight_dict.items(), key=lambda item: item[1], reverse = True)}
#
# count = 0
# for k,v in sort_dict.items():
#     if count >= 100:
#         break
#     print(f'{k} : {v}')
#     count += 1

#
from collections import Counter
from models.eda import run_stats, run_k_means

if __name__ == '__main__':
    filename = 'apis_contact.json'
    output = run_k_means(filename)
    print(list(output))
    print(Counter(output))

