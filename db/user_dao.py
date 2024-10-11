from db.database import execute_query
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


def get_user_sessions(user_id):
    query = f"SELECT * FROM user_chat_sessions WHERE user_id='{user_id}';"
    conn, cursor = execute_query(query)
    if cursor is None:
        logging.error("Failed to execute query")
        raise ValueError("Failed to execute query")
    response = cursor.fetchall()
    cursor.close()
    return response


def get_user_sessions_with_limit(user_id, limit):
    query = f"SELECT * FROM user_chat_sessions WHERE user_id='{user_id}' ORDER BY start_time desc LIMIT {limit};"
    conn, cursor = execute_query(query)
    if cursor is None:
        logging.error("Failed to execute query")
        raise ValueError("Failed to execute query")
    response = cursor.fetchall()
    cursor.close()
    return response


def get_user_history(user_id, session_id, timestamp_start, timestamp_end):
    query = f"SELECT * FROM user_chat_messages WHERE user_id='{user_id}' AND session_id='{session_id}'"
    if timestamp_start:
        query += f" AND generated_at >= '{timestamp_start}'"
    if timestamp_end:
        query += f" AND generated_at <= '{timestamp_end}'"
    conn, cursor = execute_query(query)
    if cursor is None:
        logging.error("Failed to execute query")
        raise ValueError("Failed to execute query")
    response = cursor.fetchall()
    cursor.close()
    return response


def register_message(user_id, session_id, message, type='text', images=[]):
    message_id = str(uuid.uuid4())
    images_str = ','.join(images)
    create_session_if_not_exists(user_id, session_id, message)
    query = ("INSERT INTO user_chat_messages (user_id, session_id, message_id, message, type, images) "
             "VALUES (%s, %s, %s, %s, %s, %s)")
    params = (user_id, session_id, message_id, message, type, images_str)
    conn, cursor = execute_query(query, params)
    if conn is not None:
        conn.commit()
        cursor.close()
    else:
        logging.error("Failed to execute query")


def create_session_if_not_exists(user_id, session_id, title='Title'):
    try:
        query = (f"INSERT INTO user_chat_sessions (user_id, session_id, last_message, title) "
                 f"VALUES ('{user_id}', '{session_id}', '{title}', '{title}') "
                 f" ON CONFLICT(session_id) DO UPDATE SET last_message = '{title}'")
        conn, cursor = execute_query(query)
        if cursor is None:
            logging.error("Failed to execute query")
            raise ValueError("Failed to execute query")
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logging.error(f'Already exists for user {user_id}: {e}')
