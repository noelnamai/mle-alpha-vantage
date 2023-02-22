import os
import re
import time
import nltk
import pytz
import logging
import requests
import secret_keys
import numpy as np
import pandas as pd

from io import StringIO
from airflow import DAG
from random import randint
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from airflow.hooks.S3_hook import S3Hook
from airflow.operators.python import PythonOperator

logging.basicConfig(level=logging.ERROR)

verbs_dict = {
    "edgedup": "",
    "advanced": "",
    "increased": "",
    "rose": "",
    "changed": "",
    "unchanged": "",
    "edgeddown": "-",
    "decreased": "-",
    "fell": "-",
    "declined": "-"
}

model = {
    "finished": "JJ",
    "1982" : "CPI",
    "84=100": "CPI",
    "current": "ECON",
    "dollar": "ECON",
    "gdp": "GDP",
    "quarter": "QUAT",
    "nonfarm": "ECON", 
    "payroll": "ECON", 
    "employment": "ECON", 
    "unemployment": "ECON", 
    "consumer": "ECON",
    "producer": "ECON", 
    "price": "ECON", 
    "index": "ECON"
}

nltk.download(["words", "punkt", "maxent_ne_chunker", "averaged_perceptron_tagger"], quiet=True)
treebank_tagger = nltk.data.load("taggers/maxent_treebank_pos_tagger/english.pickle")
tagger = nltk.tag.UnigramTagger(model=model, backoff=treebank_tagger)
s3_hook = S3Hook(aws_conn_id="aws-alphavantage-api")

default_args = {
    "owner": "noelnamai",
    "start_date": datetime(2022, 1, 1, 0, 0, 0, tzinfo=pytz.utc),
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

# instantiate a directed acyclic graph
dag = DAG(
    dag_id="capstone",
    default_args=default_args,
    schedule=timedelta(minutes=30),
    catchup=False,
    tags=["bls, bea, alphavantage"]
)

# function to check if day is a business day
def is_business_day(date):
    return bool(len(pd.bdate_range(date, date)))


def get_press_release_date(text: str):
    date_regex = r"([A-Za-z]+ [0-9]+, [0-9]+)"
    date_match = re.search(date_regex, text.replace(".", ""))
    date = datetime.strptime(date_match.group(1), "%B %d, %Y")
    return date


def get_bea_press_releases(indicator, params, **kwargs) -> None:
    press_releases = list()
    start_date = datetime.strptime("2010", "%Y").date()
    press_release_date = datetime.now().date()

    while press_release_date >= start_date:
        response = requests.get(url="https://www.bea.gov/news/archive?", params=params)
        soup = BeautifulSoup(response.content, "html.parser")
        press_release_elements = soup.find_all("tr", {"class": "release-row"})

        for element in press_release_elements:
            link_element = element.find("a")
            date_element = element.find("td", {"class": "views-field-created"})
            press_release_date = datetime.strptime(date_element.text.strip(), "%B %d, %Y").date()
            
            if press_release_date >= start_date:
                link = link_element["href"]
                press_releases.append(link)

        params["page"] += 1

    df = pd.DataFrame(press_releases, columns=["press_release"])
    df.to_csv(f"s3://mle-capstone-bucket/data/raw/{indicator}-press-releases.csv", index=False)


def get_bls_press_releases(indicator: str):
    press_releases = list()
    start_date = datetime.strptime("2010", "%Y").date()
    press_release_date = datetime.now().date()
    response = requests.get(f"https://www.bls.gov/bls/news-release/{indicator}.htm")
    soup = BeautifulSoup(response.text, "html.parser")
    news_releases = soup.find_all("a", href=re.compile(r"/news.release/archives/.*?.htm"))

    for item in news_releases:
        year_element = re.findall(r"\b\d{4}\b", item.text)[0]
        press_release_date = datetime.strptime(year_element.strip(), "%Y").date()

        if press_release_date >= start_date:
            link = item["href"]
            press_releases.append(link)

    df = pd.DataFrame(press_releases, columns=["press_release"])
    df.to_csv(f"s3://mle-capstone-bucket/data/raw/{indicator}-press-releases.csv", index=False)


# function to download spy daily data
def get_spy_daily_data(params, **kwargs) -> None:
    os.environ["no_proxy"] = "*"
    s3_hook.get_conn()
    params["apikey"] = secret_keys.alphavantage_key

    try:
        results = requests.get(url="https://www.alphavantage.co/query", params=params)
        data = results.json()
        data = [{"date": key, "open": value["1. open"], "close": value["4. close"], "volume": value["6. volume"]} for key, value in data["Time Series (Daily)"].items()]
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        string_data = df.to_csv(header=True, index=False)
        s3_hook.load_string(
            string_data=string_data,
            key="data/raw/spy.csv",
            bucket_name="mle-capstone-bucket",
            replace=True
        )
    except Exception as e:
        logging.error(e)
    time.sleep(30)


gdp_press_releases = PythonOperator(
    task_id="get_gdp_press_releases",
    python_callable=get_bea_press_releases,
    op_kwargs={
        "indicator": "gdp",
        "params": {
            "page": 0,
            "created_1": "All",
            "field_related_product_target_id": "451"
        }
    },
    dag=dag
)

nonfarm_payroll_press_releases = PythonOperator(
    task_id="get_nonfarm_payroll_press_releases",
    python_callable=get_bls_press_releases,
    op_kwargs={
        "indicator": "nonfarm-payroll"
    },
    dag=dag
)

gdp_press_releases >> nonfarm_payroll_press_releases
