import logging
import psycopg2
import os
from psycopg2 import sql

# Configure logging
logging.basicConfig(level=logging.INFO)

def get_db_connection():
    try:
        # Replace with your actual database connection parameters
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        return conn
    except Exception as e:
        logging.error("Error connecting to the database")
        logging.error(e)
        return None

def execute_query(query, params=None):
    try:
        conn = get_db_connection()
        if conn is None:
            raise ValueError("Failed to connect to the database")
        cursor = conn.cursor()
        cursor.execute(query, params)
        return (conn, cursor)
    except Exception as e:
        logging.error(f"Error executing query: {query} with params: {params}")
        logging.error(e)
        return (None, None)