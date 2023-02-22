from model import predict
from pydantic import BaseModel
from fastapi import FastAPI, Query, HTTPException

app = FastAPI()

class ValuesIn(BaseModel):
    nonfarm_payroll: float
    unemployment_rate: float
    producer_price_index: float
    consumer_price_index: float
    gross_domestic_product: float
    open_price: float

class ValuesOut(ValuesIn):
    forecast: int

@app.post("/predict", response_model=ValuesOut, status_code=200)
def get_prediction(payload: ValuesIn):
    nonfarm_payroll = payload.nonfarm_payroll
    unemployment_rate = payload.unemployment_rate
    producer_price_index = payload.producer_price_index
    consumer_price_index = payload.consumer_price_index
    gross_domestic_product = payload.gross_domestic_product
    open_price = payload.open_price

    prediction = predict(
        nonfarm_payroll,
        unemployment_rate,
        producer_price_index,
        consumer_price_index,
        gross_domestic_product,
        open_price,
    )

    if not prediction:
        raise HTTPException(
            status_code=400, detail="Model not found! Please train the model first."
        )

    return {
        "nonfarm_payroll": nonfarm_payroll,
        "unemployment_rate": unemployment_rate,
        "producer_price_index": producer_price_index,
        "consumer_price_index": consumer_price_index,
        "gross_domestic_product": gross_domestic_product,
        "open_price": open_price,
        "forecast": prediction,
    }
