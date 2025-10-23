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


def connect_db():
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME
        )
        print("Successfully connected with MySQL")
        return connection
    except Error as e:
        print(f"MySQL connection error: {e}")
        return None

def close_db(connection):
    if connection and connection.is_connected():
        connection.close()
        print("Closed connection MySQL")

