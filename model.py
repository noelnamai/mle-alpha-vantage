import pickle
import datetime
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

BASE_DIR = Path(__file__).resolve(strict=True).parent
TODAY = datetime.date.today()

def train(X_train, y_train):
    model = GradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=6,
        max_features=0.55,
        min_samples_leaf=12,
        min_samples_split=17,
        n_estimators=100,
        subsample=0.8,
    )
    model.fit(X_train, y_train)
    pickle.dump(model, open("model.pkl", "wb"))

def predict(nonfarm_payroll, unemployment_rate, producer_price_index, consumer_price_index, gross_domestic_product, open_price):
    model = pickle.load(open("model.pkl", "rb"))
    X_test = np.array([nonfarm_payroll, unemployment_rate, producer_price_index, consumer_price_index, gross_domestic_product, open_price]).reshape(1, -1)
    print(X_test)
    y_test = model.predict(X_test)
    return y_test[0]

if __name__ == "__main__":
    data = pd.read_csv("s3://mle-capstone-bucket/data/final/economic-data.csv", index_col="date")
    X_train = data.drop(columns=["volume", "close", "direction"]).to_numpy()
    y_train = data["direction"].to_numpy()
    train(X_train, y_train)
