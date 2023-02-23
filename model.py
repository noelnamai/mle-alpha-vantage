import pickle
import logging
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

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
    try:
        model = pickle.load(open("model.pkl", "rb"))
        X_test = np.array([nonfarm_payroll, unemployment_rate, producer_price_index, consumer_price_index, gross_domestic_product, open_price]).reshape(1, -1)
        y_test = model.predict(X_test)
    except:
        return 0
    else:
        return y_test[0]

if __name__ == "__main__":
    data = pd.read_csv("s3://mle-capstone-bucket/data/final/economic-data.csv", index_col="date")
    X_train = data.drop(columns=["volume", "close", "direction"]).to_numpy()
    y_train = data["direction"].to_numpy()
    train(X_train, y_train)
