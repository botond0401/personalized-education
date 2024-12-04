"""
Assistments Data Processing Script

This script processes the Assistments dataset for Bayesian Knowledge Tracing (BKT) or other 
educational analytics tasks. It includes functions to clean, filter, and transform the data, 
generate relevant statistics, and save the processed results.

Main Features:
- Filters data based on sequence length and minimum thresholds for problems and skills.
- Converts the dataset into a format suitable for BKT models.
- Saves the processed data and skill dictionary for further analysis.

Functions:
- clean_assistments_data: Cleans and processes the raw dataset.
- return_assistments_dict_bkt: Converts processed data into a BKT-compatible format.
- _save_dataframe: Helper function to save DataFrames and log their status.
"""

import os
import json
from collections import defaultdict
from typing import Dict, List, Literal

import pandas as pd


def clean_assistments_data(
    input_file: str,
    max_user_sequence_len: int,
    min_user_sequence_len: int,
    min_answers_per_problem: int,
    min_user_per_skill: int,
    min_len_longest_seq: int,
    num_steps: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    Cleans and preprocesses the Assistments dataset.

    Args:
        input_file (str): Path to the input CSV file.
        max_user_sequence_len (int): Maximum sequence length per user.
        min_user_sequence_len (int): Minimum sequence length per user.
        min_answers_per_problem (int): Minimum answers required per problem.
        min_user_per_skill (int): Minimum users per skill.
        min_len_longest_seq (int): Minimum length of the longest user sequence per skill.
        num_steps (int): How many times to do all the checks.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Processed dataframes for problems and skills.
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
    df_data_filtered = df_data_filtered.dropna(subset=['skill_id']).copy()
    df_data_filtered['bottom_hint'] = df_data_filtered['bottom_hint'].fillna(0)

    # Create the 'timestamp' column
    df_data_filtered['opportunity_padded'] = df_data_filtered['opportunity'].apply(
        lambda x: f"{int(x):04d}"
    )
    df_data_filtered['timestamp'] = (
        df_data_filtered['order_id'].astype(str) + df_data_filtered['opportunity_padded']
    )
    df_data_filtered.drop(columns=['opportunity', 'opportunity_padded', 'order_id'], inplace=True)

    # Type conversion
    df_data_filtered['skill_id'] = pd.to_numeric(df_data_filtered['skill_id'], errors='coerce').astype('Int64')
    for col in ['correct', 'ms_first_response', 'bottom_hint']:
        df_data_filtered[col] = df_data_filtered[col].astype(int)
    for col in ['user_id', 'skill_id', 'skill_name', 'problem_id', 'timestamp']:
        df_data_filtered[col] = df_data_filtered[col].astype(str)

    # Sort by user_id and timestamp
    df_data_sorted = df_data_filtered.sort_values(by=['user_id', 'timestamp'])

    # Prepare separate dataframes for answers and skills
    df_skills_basic = df_data_sorted[['skill_id', 'skill_name', 'problem_id']].drop_duplicates()
    df_answers_basic = df_data_sorted.drop(columns=['skill_id', 'skill_name']).drop_duplicates()

    for _ in range(num_steps):
        # Filter answers by user sequence length
        df_answers_valid = (
            df_answers_basic.groupby('user_id')
            .filter(lambda x: len(x) >= min_user_sequence_len) # a user has to have at least 'min_user_sequence_len' answers
            .groupby('user_id')
            .head(max_user_sequence_len) # a user can have at most 'max_user_sequence_len' answers
        )

        # Filter problems by minimum answers
        df_answers_valid = df_answers_valid.groupby('problem_id').filter(
            lambda x: len(x) >= min_answers_per_problem # an answer has to appear at least 'min_answers_per_problem' times
        )

        # Filter skills by valid problems
        valid_problem_ids = df_answers_valid['problem_id']
        df_skills_valid = df_skills_basic[df_skills_basic['problem_id'].isin(valid_problem_ids)]

        # Filter skills by user counts
        df_skills_merged = pd.merge(df_answers_valid, df_skills_valid, on='problem_id')
        skill_user_counts = df_skills_merged.groupby('skill_id')['user_id'].nunique()
        valid_skills = skill_user_counts[skill_user_counts >= min_user_per_skill].index
        df_skills_valid = df_skills_valid[df_skills_valid['skill_id'].isin(valid_skills)]

        # Filter skills by user sequence length
        user_skill_counts = df_skills_merged.groupby(['skill_id', 'user_id']).size().reset_index(name='user_count')
        df_max_user_count_per_skill = user_skill_counts.groupby('skill_id')['user_count'].max()
        valid_skills = df_max_user_count_per_skill[df_max_user_count_per_skill > min_len_longest_seq].index

        df_skills_basic = df_skills_valid[df_skills_valid['skill_id'].isin(valid_skills)]

        # Filter problems again based on skills
        df_answers_basic = df_answers_valid[df_answers_valid['problem_id'].isin(df_skills_basic['problem_id'])]

    return df_answers_basic, df_skills_basic


def return_assistments_dict_bkt(
    df_assistments_answers: pd.DataFrame,
    df_assistment_skills: pd.DataFrame
) -> Dict[str, List[List[Literal[0, 1]]]]:
    """
    Converts Assistments data into a Bayesian Knowledge Tracing (BKT) dictionary.

    Groups data by skill ID and user ID, creating a structure where each skill maps 
    to lists of user answer sequences.

    Args:
        df_assistments_answers (pd.DataFrame): Processed answers data, including user responses.
        df_assistment_skills (pd.DataFrame): Processed skills data, including problem-skill mappings.

    Returns:
        Dict[str, List[List[Literal[0, 1]]]]:
            A dictionary where keys are skill IDs, and values are lists of answer sequences.
            Each sequence corresponds to a user and contains answers as 0s and 1s.
    """
    # Merge answers with skills to align skill IDs with answers
    df_data = pd.merge(df_assistments_answers, df_assistment_skills, on='problem_id', how='inner')

    # Create a defaultdict to collect answer sequences per skill
    skill_dict = defaultdict(list)

    # Group the DataFrame by skill_id and user_id
    for skill_id, skill_group in df_data.groupby('skill_id'):
        answer_list = []
        for _, user_group in skill_group.groupby('user_id'):
            # Extract the list of answers for this user
            answers = user_group['correct'].tolist()
            answer_list.append(answers)
    
        skill_dict[skill_id] = answer_list

    # Convert defaultdict to a regular dictionary and return
    return dict(skill_dict)


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
    print(f"{description} saved to {output_path}.")


if __name__ == "__main__":
    # Constants
    INPUT_FILE = 'data/raw/skill_builder_data.csv'
    OUTPUT_FOLDER = 'data/preprocessed/'
    OUTPUT_FILE_ANSWERS_DF = 'answers_df.csv'
    OUTPUT_FILE_ANSWER_SKILL_MAPPING_DF = 'answer_skill_mapping_df.csv'
    OUTPUT_FILE_SKILLS_D = 'skills_dict.json'

    MAX_USER_SEQUENCE_LEN = 400
    MIN_USER_SEQUENCE_LEN = 5
    MIN_ANSWERS_PER_PROBLEM = 10
    MIN_USERS_PER_SKILL = 10
    MIN_LEN_LONGEST_SEQ = 3
    # reasoning behind the choices of these values can be found in 'notebooks/data_exploration.ipynb'

    # Clean and preprocess the data
    df_answers, df_skills = clean_assistments_data(
        input_file=INPUT_FILE,
        max_user_sequence_len=MAX_USER_SEQUENCE_LEN,
        min_user_sequence_len=MIN_USER_SEQUENCE_LEN,
        min_answers_per_problem=MIN_ANSWERS_PER_PROBLEM,
        min_user_per_skill=MIN_USERS_PER_SKILL,
        min_len_longest_seq=MIN_LEN_LONGEST_SEQ,
    )

    # Summarize the data
    summary = {
        "Number of users": df_answers['user_id'].nunique(),
        "Number of problems": df_skills['problem_id'].nunique(),
        "Number of skills": df_skills['skill_id'].nunique(),
        "Number of answers": len(df_answers),
    }

    # Print summary
    print("\nData Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    # Save processed data
    _save_dataframe(df_answers, OUTPUT_FOLDER, OUTPUT_FILE_ANSWERS_DF, "Answer dataframe")
    _save_dataframe(df_skills, OUTPUT_FOLDER, OUTPUT_FILE_ANSWER_SKILL_MAPPING_DF, "Skill dataframe")

    # Generate and save the skill dictionary for BKT
    skill_d = return_assistments_dict_bkt(df_answers, df_skills)
    OUTPUT_PATH_SKILLS_D = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE_SKILLS_D)
    with open(OUTPUT_PATH_SKILLS_D, 'w', encoding='utf-8') as json_file:
        json.dump(skill_d, json_file)
    print(f"Skill dictionary saved to {OUTPUT_PATH_SKILLS_D}.")
