import codecs
import csv
import json
from io import StringIO

import pandas as pd
import uvicorn

from models.model_generator import generator
from models.no_decomposition_model import ModelNoDecomposition
from fastapi import FastAPI, File, UploadFile, status, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, RedirectResponse

app = FastAPI(title="Bumble Bee", description="Predict LD50 value by linear regression.")


@app.post("/upload/")
def upload_csv(decomposition: str = Query(...), csv_file: UploadFile = File(...)):
    # return {'file':csv_file.filename}
    # data = csv.reader(codecs.iterdecode(csv_file.file,'utf-8'), delimiter='\t')

    dataframe = pd.read_json(csv_file.file, encoding='utf-8')
    # dataframe = pd.DataFrame([row for row in csv_reader])

    # dataframe = pd.read_csv(csv_file.filename, encoding='utf-8')
    try:
        model = generator().create_model(decomposition, dataframe)

        r_list = model.train()
        output = {f'fold_{idx+1}': item for idx, item in enumerate(r_list)}
        return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(output))
    except Exception as e:
        output = {
            'error message' : e
        }
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=jsonable_encoder(output))

@app.get("/")
def index():
    return RedirectResponse('/docs')

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080, debug=True)