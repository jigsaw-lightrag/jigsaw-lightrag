from urllib import parse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os



load_dotenv()

server=os.getenv("db_server")
database=os.getenv("database")
username=os.getenv("db_username")
password=os.getenv("db_password")

import threading
db_write_lock = threading.Lock()

def get_engine():
    db_path = f"mssql+pyodbc://{username}:{parse.quote_plus(password)}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
    return create_engine(db_path, echo=False)

def get_db():
    db = get_sessionmaker()()
    try:
        yield db
    finally:
        db.close()

def get_sessionmaker():
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)


    
