import pandas as pd
import uvicorn

from models.no_decomposition_model import ModelNoDecomposition
from fastapi import FastAPI, File, UploadFile, status, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

app = FastAPI(title="Bumble Bee", description="Predict LD50 value by linear regression.")

@app.post("/upload/")
def upload_csv(test_size: float = Query(...),
               csv_file: UploadFile = File(...)):
    dataframe = pd.read_csv(csv_file.file)
    try:
        m = ModelNoDecomposition(dataframe, test_size)
        cod, mse = m.run()
        output = {
            'Coefficient of determination' : cod,
            'Mean square error' : mse
        }
        return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(output))
    except Exception as e:
        output = {
            'error message' : e
        }
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=jsonable_encoder(output))

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', debug=True)