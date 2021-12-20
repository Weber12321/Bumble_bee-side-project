import codecs
import csv
import json
from enum import Enum
from io import StringIO
from typing import List, Dict

import pandas as pd
import uvicorn

from models.model_generator import generator
from models.no_decomposition_model import ModelNoDecomposition
from fastapi import FastAPI, File, UploadFile, status, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, RedirectResponse

from utils.helper import get_logger

app = FastAPI(title="Bumble Bee", description="Predict LD50 value by linear regression.")

class Decomposition(str, Enum):
    no_doc = 'no_decomposition'
    doc = 'decomposition'


@app.post("/upload/")
def upload_csv(decomposition: Decomposition, csv_file: UploadFile = File(...)):
    # return {'file':csv_file.filename}
    # data = csv.reader(codecs.iterdecode(csv_file.file,'utf-8'), delimiter='\t')
    dataframe = pd.read_json(csv_file.file, encoding='utf-8')
    # dataframe = pd.DataFrame([row for row in csv_reader])

    # dataframe = pd.read_csv(csv_file.filename, encoding='utf-8')

    model = generator().create_model(decomposition, dataframe)

    r = model.train()
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(r))


@app.get("/")
def index():
    return RedirectResponse('/docs')

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080, debug=True)