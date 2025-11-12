import mysql.connector
import os
from dotenv import load_dotenv
from mysql.connector import Error

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")

_connection = None

def get_connection():
    global _connection
    if _connection is None or not _connection.is_connected():
        try:
            _connection = mysql.connector.connect(
                host=DB_HOST,
                port=DB_PORT,
                user=DB_USER,
                password=DB_PASS,
                database=DB_NAME
            )
            print("Successfully connected with MySQL")
        except Error as e:
            print(f"MySQL connection error: {e}")
            _connection = None
    return _connection

def close_connection():
    global _connection
    if _connection and _connection.is_connected():
        _connection.close()
        print("Closed MySQL connection")
        _connection = None
