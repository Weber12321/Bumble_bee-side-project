import codecs
import csv
from io import StringIO

import pandas as pd
import uvicorn

from models.no_decomposition_model import ModelNoDecomposition
from fastapi import FastAPI, File, UploadFile, status, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, RedirectResponse

app = FastAPI(title="Bumble Bee", description="Predict LD50 value by linear regression.")

@app.get("/")
def index():
    return RedirectResponse('/docs')


@app.post("/upload/")
def upload_csv(csv_file: UploadFile = File(...)):
    # data = csv.reader(codecs.iterdecode(csv_file.file,'utf-8'), delimiter='\t')

    # dataframe = pd.read_csv(StringIO(csv_file), encoding='utf-8')
    dataframe = pd.read_csv(csv_file.filename, encoding='utf-8', low_memory=False)
    try:
        model = ModelNoDecomposition(dataframe)
        r_list = model.train()
        output = {f'fold_{idx+1}': item for idx, item in enumerate(r_list)}
        return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(output))
    except Exception as e:
        output = {
            'error message' : e
        }
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=jsonable_encoder(output))

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', debug=True)