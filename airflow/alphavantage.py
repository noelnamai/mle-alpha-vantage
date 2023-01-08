import os
import time
import pytz
import logging
import requests
import secret_keys
import numpy as np
import pandas as pd

from io import StringIO
from airflow import DAG
from datetime import datetime, timedelta
from airflow.hooks.S3_hook import S3Hook
from airflow.operators.python import PythonOperator

logging.basicConfig(level=logging.ERROR)

s3_hook = S3Hook(aws_conn_id="aws-alphavantage-api")

default_args = {
    "owner": "noelnamai",
    "start_date": datetime(2022, 1, 1, 0, 0, 0, tzinfo=pytz.utc),
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

# instantiate a directed acyclic graph
dag = DAG(
    dag_id="alphavantage",
    default_args=default_args,
    schedule=timedelta(minutes=60),
    catchup=False,
    tags=["alphavantage"]
)

# function to check if day is a business day
def is_business_day(date):
    return bool(len(pd.bdate_range(date, date)))

# function to download data from the alphavantage api
def download_alphavantage_api_data(var_name, params, **kwargs) -> None:
    os.environ["no_proxy"] = "*"
    s3_hook.get_conn()
    params["apikey"] = secret_keys.alphavantage_key
    try:
        results = requests.get(
            url="https://www.alphavantage.co/query",
            params=params
        )
        data = results.json()
        df = pd.DataFrame(data["data"])
        df.columns = ["date", var_name]
        df["date"] = pd.to_datetime(df["date"])
        string_data = df.to_csv(header=True, index=False)
        s3_hook.load_string(
            string_data=string_data,
            key=f"data/raw/{var_name}.csv",
            bucket_name="airflow-alphavantage-bucket",
            replace=True
        )
    except Exception as e:
        logging.error(e)
    time.sleep(30)

# function to download top five spy company earnings
def get_company_earnings(params, **kwargs) -> None:
    os.environ["no_proxy"] = "*"
    s3_hook.get_conn()
    for company in ["AAPL", "MSFT", "AMZN", "TSLA", "GOOGL"]:
        params["symbol"] = company
        params["apikey"] = secret_keys.alphavantage_key
        try:
            results = requests.get(
                url="https://www.alphavantage.co/query",
                params=params
            )
            data = results.json()
            annual_df = pd.DataFrame(data["annualEarnings"])
            annual_df.columns = ["date", company.lower() + "_annual_eps"]
            annual_df["date"] = pd.to_datetime(annual_df["date"])
            quarterly_df = pd.DataFrame(data["quarterlyEarnings"])
            quarterly_df = quarterly_df[["reportedDate", "reportedEPS", "surprise"]]
            quarterly_df.columns = ["date", company.lower() + "_quarterly_eps", company.lower() + "_surprise"]
            quarterly_df["date"] = pd.to_datetime(quarterly_df["date"])
            df = pd.merge(annual_df, quarterly_df, how="outer")
            string_data = df.to_csv(header=True, index=False)
            s3_hook.load_string(
                string_data=string_data,
                key=f"data/raw/{company.lower()}_earnings.csv",
                bucket_name="airflow-alphavantage-bucket",
                replace=True
            )
        except Exception as e:
            logging.error(e)
        time.sleep(30)

# function to download spy daily data
def get_spy_daily_data(params, **kwargs) -> None:
    os.environ["no_proxy"] = "*"
    s3_hook.get_conn()
    params["apikey"] = secret_keys.alphavantage_key
    try:
        results = requests.get(
            url="https://www.alphavantage.co/query",
            params=params
        )
        data = results.json()
        data = [{"date": key, "open": value["1. open"], "close": value["4. close"], "volume": value["6. volume"]} for key, value in data["Time Series (Daily)"].items()]
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        string_data = df.to_csv(header=True, index=False)
        s3_hook.load_string(
            string_data=string_data,
            key="data/raw/spy.csv",
            bucket_name="airflow-alphavantage-bucket",
            replace=True
        )
    except Exception as e:
        logging.error(e)
    time.sleep(30)

def clean_api_data(**kwargs) -> None:
    os.environ["no_proxy"] = "*"
    s3_hook.get_conn()
    try:
        s3_keys = s3_hook.list_keys(
            prefix="data/raw/",
            bucket_name="airflow-alphavantage-bucket"
        )
        df = pd.concat([pd.read_csv(StringIO(s3_hook.read_key(bucket_name="airflow-alphavantage-bucket", key=key))) for key in s3_keys])
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"].apply(is_business_day)]
        df.sort_values(by="date", inplace=True, ignore_index=True)
        df.replace({".": np.nan, "None": np.nan}, inplace=True)
        df.fillna(method="ffill", inplace=True)
        numeric_columns = df.columns.drop(["date"])
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
        df.drop_duplicates(inplace=True, keep="last", ignore_index=True)
        string_data = df.to_csv(header=True, index=False)
        s3_hook.load_string(
            string_data=string_data,
            key="data/final/alphavantage.csv",
            bucket_name="airflow-alphavantage-bucket",
            replace=True
        )
    except Exception as e:
        logging.error(e)
    time.sleep(30)

def create_new_data_features(**kwargs) -> None:
    os.environ["no_proxy"] = "*"
    s3_hook.get_conn()
    try:
        df = pd.read_csv(StringIO(s3_hook.read_key(bucket_name="airflow-alphavantage-bucket", key="data/final/alphavantage.csv")))
        df["date"] = pd.to_datetime(df["date"])
        df["day"] = df["date"].dt.day_name()
        df["month"] = df["date"].dt.month_name()
        df["year"] = df["date"].dt.year
        df = df[df["date"] >= "2002-01-01"]
        string_data = df.to_csv(header=True, index=False)
        s3_hook.load_string(
            string_data=string_data,
            key="data/final/alphavantage.csv",
            bucket_name="airflow-alphavantage-bucket",
            replace=True
        )
    except Exception as e:
        logging.error(e)
    time.sleep(30)


gdp = PythonOperator(
    task_id="get_gdp_data",
    python_callable=download_alphavantage_api_data,
    op_kwargs={
        "var_name": "gross_domestic_product",
        "params": {
            "function": "REAL_GDP",
            "interval": "quarterly"
        }
    },
    dag=dag
)

gdp_per_capita = PythonOperator(
    task_id="get_gdp_per_capita_data",
    python_callable=download_alphavantage_api_data,
    op_kwargs={
        "var_name": "gdp_per_capita",
        "params": {
            "function": "REAL_GDP_PER_CAPITA"
        }
    },
    dag=dag
)

two_year_treasury_yield = PythonOperator(
    task_id="get_two_year_treasury_yield_data",
    python_callable=download_alphavantage_api_data,
    op_kwargs={
        "var_name": "2yr_treasury_yield",
        "params": {
            "function": "TREASURY_YIELD",
            "interval": "daily",
            "maturity": "2year"
        }
    },
    dag=dag
)

five_year_treasury_yield = PythonOperator(
    task_id="get_five_year_treasury_yield_data",
    python_callable=download_alphavantage_api_data,
    op_kwargs={
        "var_name": "5yr_treasury_yield",
        "params": {
            "function": "TREASURY_YIELD",
            "interval": "daily",
            "maturity": "5year"
        }
    },
    dag=dag
)

ten_year_treasury_yield = PythonOperator(
    task_id="get_ten_year_treasury_yield_data",
    python_callable=download_alphavantage_api_data,
    op_kwargs={
        "var_name": "10yr_treasury_yield",
        "params": {
            "function": "TREASURY_YIELD",
            "interval": "daily",
            "maturity": "10year"
        }
    },
    dag=dag
)

thirty_year_treasury_yield = PythonOperator(
    task_id="get_thirty_year_treasury_yield_data",
    python_callable=download_alphavantage_api_data,
    op_kwargs={
        "var_name": "30yr_treasury_yield",
        "params": {
            "function": "TREASURY_YIELD",
            "interval": "daily",
            "maturity": "30year"
        }
    },
    dag=dag
)

federal_interest_rate_data = PythonOperator(
    task_id="get_federal_interest_rate_data",
    python_callable=download_alphavantage_api_data,
    op_kwargs={
        "var_name": "federal_interest_rate",
        "params": {
            "function": "FEDERAL_FUNDS_RATE",
            "interval": "daily"
        }
    },
    dag=dag
)

consumer_price_index_data = PythonOperator(
    task_id="get_consumer_price_index_data",
    python_callable=download_alphavantage_api_data,
    op_kwargs={
        "var_name": "consumer_price_index",
        "params": {
            "function": "CPI",
            "interval": "monthly"
        }
    },
    dag=dag
)

inflation_rate_data = PythonOperator(
    task_id="get_inflation_rate_data",
    python_callable=download_alphavantage_api_data,
    op_kwargs={
        "var_name": "inflation_rate",
        "params": {
            "function": "INFLATION"
        }
    },
    dag=dag
)

inflation_expectation_data = PythonOperator(
    task_id="get_inflation_expectation_data",
    python_callable=download_alphavantage_api_data,
    op_kwargs={
        "var_name": "inflation_expectation",
        "params": {
            "function": "INFLATION_EXPECTATION"
        }
    },
    dag=dag
)

consumer_sentiment_data = PythonOperator(
    task_id="get_consumer_sentiment_data",
    python_callable=download_alphavantage_api_data,
    op_kwargs={
        "var_name": "consumer_sentiment",
        "params": {
            "function": "CONSUMER_SENTIMENT"
        }
    },
    dag=dag
)

retail_sales_data = PythonOperator(
    task_id="get_retail_sales_data",
    python_callable=download_alphavantage_api_data,
    op_kwargs={
        "var_name": "retail_sales",
        "params": {
            "function": "RETAIL_SALES"
        }
    },
    dag=dag
)

manufacturers_orders_data = PythonOperator(
    task_id="get_manufacturers_orders_data",
    python_callable=download_alphavantage_api_data,
    op_kwargs={
        "var_name": "manufacturers_orders",
        "params": {
            "function": "DURABLES"
        }
    },
    dag=dag
)

unemployment_data = PythonOperator(
    task_id="get_unemployment_data",
    python_callable=download_alphavantage_api_data,
    op_kwargs={
        "var_name": "unemployment",
        "params": {
            "function": "UNEMPLOYMENT"
        }
    },
    dag=dag
)

nonfarm_payroll_data = PythonOperator(
    task_id="get_nonfarm_payroll_data",
    python_callable=download_alphavantage_api_data,
    op_kwargs={
        "var_name": "nonfarm_payroll",
        "params": {
            "function": "NONFARM_PAYROLL"
        }
    },
    dag=dag
)

company_earnings = PythonOperator(
    task_id=f"get_company_earnings",
    python_callable=get_company_earnings,
    op_kwargs={
        "params": {
            "function": "EARNINGS"
        }
    },
    dag=dag
)

spy_daily_data = PythonOperator(
    task_id="get_spy_daily_data",
    python_callable=get_spy_daily_data,
    op_kwargs={
        "params": {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": "SPY",
            "outputsize": "full"
        }
    },
    dag=dag
)

clean_data = PythonOperator(
    task_id="clean_api_data",
    python_callable=clean_api_data,
    dag=dag
)

feature_engineering = PythonOperator(
    task_id="create_new_data_features",
    python_callable=create_new_data_features,
    dag=dag
)

gdp >> gdp_per_capita >> [two_year_treasury_yield, five_year_treasury_yield, ten_year_treasury_yield, thirty_year_treasury_yield] >> federal_interest_rate_data >> [consumer_price_index_data, inflation_rate_data, inflation_expectation_data] >> consumer_sentiment_data >> [retail_sales_data, manufacturers_orders_data, unemployment_data, nonfarm_payroll_data] >> company_earnings >> spy_daily_data >> clean_data >> feature_engineering
