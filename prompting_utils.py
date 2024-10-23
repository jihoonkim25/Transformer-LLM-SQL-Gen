import json
import re


def read_schema(schema_path):
    '''
    Load and parse the schema file.
    '''
    with open(schema_path, 'r') as file:
        schema = json.load(file)
    return schema


def extract_sql_query(response):
    """
    Extract the last SQL query from the model's response, from the last "SQL:" until "<eos>".

    Args:
        response (str): The response from the model which contains one or more SQL queries.

    Returns:
        str: The extracted SQL query, ensuring it is a single line.
    """
    # Find the last occurrence of "SQL:"
    last_sql_index = response.rfind("SQL:")
    if last_sql_index == -1:
        return "No SQL query found in the response"

    # Find the index of "<eos>" after the last occurrence of "SQL:"
    eos_index = response.find("<eos>", last_sql_index)
    if eos_index == -1:
        eos_index = len(response)  # If "<eos>" is not found, take the rest of the string

    # Extract the substring from "SQL:" to "<eos>"
    sql_query = response[last_sql_index:eos_index].replace("SQL:", "").strip()

    # Replace newlines and excessive whitespace with a single space
    sql_query = ' '.join(sql_query.split())

    sql_query = sql_query.replace("```", "")
    return sql_query


def save_logs(output_path, sql_em, record_em, record_f1, error_msgs):
    """
    Save the logs of the experiment to files.
    You can change the format as needed.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(
            f"SQL EM: {sql_em}\nRecord EM: {record_em}\nRecord F1: {record_f1}\nModel Error Messages: {error_msgs}\n"
        )
