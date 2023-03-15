import pyodbc
import pandas as pd
import toml


SECRETS = toml.load("streamlit/secrets.toml")

server = SECRETS['database']['sql_server']
database = SECRETS['database']['sql_database_name']
username = SECRETS['database']['sql_username']
password = SECRETS['database']['sql_password']
driver= '{ODBC Driver 17 for SQL Server}'


def sql_to_pandas(query: str):
    with pyodbc.connect('DRIVER='+driver+';SERVER=tcp:'+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password) as conn:
        df = pd.read_sql(query, con=conn)
    return df
