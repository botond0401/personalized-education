"""
Assistments Data Processing Script

This script processes the Assistments dataset for Bayesian Knowledge Tracing (BKT) or other 
educational analytics tasks. It includes functions to clean, filter, and transform the data, 
generate relevant statistics, and save the processed results.

Main Features:
- Filters data based on sequence length and thresholds for problems and skills.
- Converts the dataset into a format suitable for BKT models or other machine learning tasks.
- Saves the processed data for further analysis or modeling.

Functions:
- clean_assistments_data: Cleans and preprocesses the raw dataset to meet analysis requirements.
- _save_dataframe: Saves DataFrames to a specified location and logs the operation.

Usage:
- Adjust the constants in the __main__ block to match your dataset and requirements.
- Ensure the input file is located at the specified path before running the script.
"""

import os
from collections import defaultdict
from typing import Literal

import pandas as pd

def clean_assistments_data(
    input_file: str,
    max_user_sequence_len: int,
    min_user_sequence_len: int,
    min_user_per_skill: int,
    min_len_longest_seq: int,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    Cleans and preprocesses the Assistments dataset.

    Args:
        input_file (str): Path to the input CSV file.
        max_user_sequence_len (int): Maximum sequence length per user to retain.
        min_user_sequence_len (int): Minimum sequence length per user to retain.
        min_user_per_skill (int): Minimum number of users required per skill.
        min_len_longest_seq (int): Minimum length of the longest sequence required for skills.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Processed DataFrames for answers and skill names.
    """
    try:
        df_data = pd.read_csv(input_file, encoding='ISO-8859-1', low_memory=False)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: File '{input_file}' not found.") from e

    # Validate required columns
    columns_to_keep = [
        'order_id', 'user_id', 'correct', 'skill_id', 'skill_name', 
        'problem_id', 'ms_first_response', 'bottom_hint', 'opportunity'
    ]
    if not all(col in df_data.columns for col in columns_to_keep):
        raise ValueError("Input file missing required columns.")
    df_data_filtered = df_data[columns_to_keep]

    # Drop rows without skill_id and fill missing values
    df_data_full = df_data_filtered.dropna(subset=['skill_id']).copy()
    df_data_full['bottom_hint'] = df_data_full['bottom_hint'].fillna(0)

    # Create a composite 'timestamp' column to order actions by users
    df_data_full['opportunity_padded'] = df_data_full['opportunity'].apply(
        lambda x: f"{int(x):04d}"
    )
    df_data_full['timestamp'] = (
        df_data_full['order_id'].astype(str) + df_data_full['opportunity_padded']
    )
    df_data_full.drop(columns=['opportunity', 'opportunity_padded', 'order_id'], inplace=True)

    # Type conversions
    df_data_full['skill_id'] = pd.to_numeric(df_data_full['skill_id'], errors='coerce').astype('Int64')
    columns_to_int = ['correct', 'ms_first_response', 'bottom_hint']
    df_data_full[columns_to_int] = df_data_full[columns_to_int].astype(int)

    # Convert specific columns to string
    columns_to_str = ['user_id', 'skill_id', 'skill_name', 'problem_id', 'timestamp']
    df_data_full[columns_to_str] = df_data_full[columns_to_str].astype(str)

    # Sort data by user and timestamp
    df_data_sorted = df_data_full.sort_values(by=['user_id', 'timestamp'])

    # Prepare separate DataFrames for mapping and analysis
    df_skill_problem_mapping = df_data_sorted[['skill_id', 'skill_name', 'problem_id']].drop_duplicates().sort_values(by='skill_id').reset_index(drop=True)
    df_answers = df_data_sorted.drop(columns=['skill_id', 'skill_name']).drop_duplicates().reset_index(drop=True)
    df_answers = df_answers.merge(df_skill_problem_mapping, on='problem_id')

    # Reorder columns in df_answers
    cols = df_answers.columns.tolist()
    cols[-1], cols[1] = cols[1], cols[-1]  # Swap second and last column positions
    df_answers = df_answers[cols]

    # Shuffle and reorder data
    df_answers = df_answers.sample(frac=1, random_state=42).reset_index(drop=True)
    df_answers = df_answers.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)

    # Create a binary column for each skill_id
    columns_to_keep = ["user_id", "timestamp", "problem_id", "ms_first_response", "bottom_hint", "correct"]
    df_answers_pivot = df_answers.pivot_table(
        index=columns_to_keep,
        columns="skill_id",
        aggfunc="size",
        fill_value=0
    ).reset_index()

    # Flatten column names and reorder by user and timestamp
    df_answers_pivot.columns.name = None
    df_answers_pivot = df_answers_pivot.rename_axis(None, axis=1)
    df_answers_pivot = df_answers_pivot.sort_values(by=['user_id', 'timestamp'])

    # Filter users based on sequence length
    df_filtered_user = df_answers_pivot.groupby('user_id').filter(lambda x: len(x) >= min_user_sequence_len)
    df_answers_valid_user = df_filtered_user.groupby('user_id').head(max_user_sequence_len)

    # Remove skills with insufficient users
    num_non_skill_columns = len(columns_to_keep)
    skill_occurrences = df_answers_valid_user.iloc[:, num_non_skill_columns:].sum()
    skills_to_remove = skill_occurrences[skill_occurrences < min_user_per_skill].index
    df_answers_valid_skills = df_answers_valid_user[[col for col in df_answers_valid_user.columns if col not in skills_to_remove]]
    df_answers_valid_skills = df_answers_valid_skills[df_answers_valid_skills.iloc[:, num_non_skill_columns:].sum(axis=1) != 0]

    # Filter skills by sequence length
    user_skill_counts = df_answers_valid_skills.groupby('user_id').sum()
    max_skill_counts = user_skill_counts.iloc[:, num_non_skill_columns-1:].max()
    skills_to_remove = max_skill_counts[max_skill_counts < min_len_longest_seq].index
    df_answers_valid_skills = df_answers_valid_skills[[col for col in df_answers_valid_skills.columns if col not in skills_to_remove]]
    df_answers_valid_skills = df_answers_valid_skills[df_answers_valid_skills.iloc[:, num_non_skill_columns:].sum(axis=1) != 0]

    valid_skill_ids = df_answers_valid_skills.columns[num_non_skill_columns:]
    df_valid_skill_names = df_skill_problem_mapping.loc[df_skill_problem_mapping['skill_id'].isin(valid_skill_ids), ['skill_id', 'skill_name']].drop_duplicates()

    return df_answers_valid_skills, df_valid_skill_names

def _save_dataframe(
        df: pd.DataFrame,
        output_folder: str,
        output_file: str,
        description: str
        ) -> None:
    """
    Saves a DataFrame to a specified folder and logs the operation.

    Args:
        df (pd.DataFrame): DataFrame to save.
        output_folder (str): Directory where the file will be saved.
        output_file (str): Name of the output file.
        description (str): Description of the data being saved for logging purposes.

    Returns:
        None
    """
    output_path = os.path.join(output_folder, output_file)
    df.to_csv(output_path, index=False)
    print(f"\n{description} saved to {output_path}.")

if __name__ == "__main__":
    # Constants
    INPUT_FILE = 'data/raw/skill_builder_data.csv'
    OUTPUT_FOLDER = 'data/preprocessed/'
    OUTPUT_FILE_DF_ANSWERS = 'df_answers.csv'
    OUTPUT_FILE_DF_SKILL_NAMES = 'df_skill_names.csv'

    MAX_USER_SEQUENCE_LEN = 400
    MIN_USER_SEQUENCE_LEN = 5
    MIN_USERS_PER_SKILL = 10
    MIN_LEN_LONGEST_SEQ = 3

    # Clean and preprocess the data
    df_answers, df_skill_names = clean_assistments_data(
        input_file=INPUT_FILE,
        max_user_sequence_len=MAX_USER_SEQUENCE_LEN,
        min_user_sequence_len=MIN_USER_SEQUENCE_LEN,
        min_user_per_skill=MIN_USERS_PER_SKILL,
        min_len_longest_seq=MIN_LEN_LONGEST_SEQ,
    )

    # Summarize the data
    summary = {
        "Number of users": df_answers['user_id'].nunique(),
        "Number of skills": len(df_skill_names),
        "Number of answers": len(df_answers),
    }

    # Print summary
    print("\nData Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    # Save processed data
    _save_dataframe(df_answers, OUTPUT_FOLDER, OUTPUT_FILE_DF_ANSWERS, "Valid answers dataframe")
    _save_dataframe(df_skill_names, OUTPUT_FOLDER, OUTPUT_FILE_DF_SKILL_NAMES, "Skill names dataframe")
