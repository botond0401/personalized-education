import json
from collections import defaultdict
from typing import Dict, Tuple, List, Literal

import pandas as pd


def clean_assistments_data(
        input_file: str,
        max_user_sequence_len: int,
        min_user_sequence_len: int,
        min_answers_per_problem: int,
        min_user_per_skill: int,
        min_len_longest_seq: int
        ) -> pd.DataFrame:
    df_data = pd.read_csv(input_file, encoding='ISO-8859-1', low_memory=False)
    columns_to_keep = ['order_id', 'user_id', 'correct', 'skill_id', 'skill_name', 'problem_id',
                                'ms_first_response', 'bottom_hint', 'opportunity']
    df_data_filtered = df_data[columns_to_keep]
    df_data_filtered.isna().sum()

    df_data_full = df_data_filtered.dropna(subset='skill_id')

    df_data_full = df_data_full.copy()

    df_data_full['bottom_hint'] = df_data_full['bottom_hint'].fillna(0)

    df_data_full['opportunity_padded'] = df_data_full['opportunity'].apply(lambda x: f"{int(x):04d}")

    # Create the 'timestamp' column by concatenating 'order_id' and padded 'opportunity'
    df_data_full['timestamp'] = df_data_full['order_id'].astype(str) + df_data_full['opportunity_padded']

    df_data_full.drop(columns=['opportunity', 'opportunity_padded', 'order_id'], inplace=True)

    df_data_full['skill_id'] = pd.to_numeric(df_data_full['skill_id'], errors='coerce').astype('Int64')
    columns_to_int = ['correct', 'ms_first_response', 'bottom_hint']  # Replace with your column names
    df_data_full[columns_to_int] = df_data_full[columns_to_int].astype(int)

    # Convert specific columns to string
    columns_to_str = ['user_id', 'skill_id', 'skill_name', 'problem_id', 'timestamp']  # Replace with your column names
    df_data_full[columns_to_str] = df_data_full[columns_to_str].astype(str)

    df_data_sorted = df_data_full.sort_values(by=['user_id', 'timestamp'])

    df_skills = df_data_sorted[['skill_id', 'skill_name', 'problem_id']].drop_duplicates().sort_values(by='skill_id').reset_index(drop=True)
    df_answers = df_data_sorted.drop(columns=['skill_id', 'skill_name']).drop_duplicates().reset_index(drop=True)

    filtered_df = df_answers.groupby('user_id').filter(lambda x: len(x) >= min_user_sequence_len)

    # Then, for each user_id, keep only the first max_user_sequence_len rows
    df_answers_valid = filtered_df.groupby('user_id').head(max_user_sequence_len)

    df_problems_valid = df_answers_valid.groupby('problem_id').filter(lambda x: len(x) >= min_answers_per_problem)

    df_skills = df_skills[df_skills['problem_id'].isin(df_problems_valid['problem_id'].unique())]

    df_skills_merged = pd.merge(df_problems_valid, df_skills, on='problem_id')

    df_skills_per_user = df_skills_merged.groupby('skill_id')['user_id'].nunique()
    list_skills_to_drop = df_skills_per_user[df_skills_per_user < min_user_per_skill].index

    df_skills = df_skills[~df_skills['skill_id'].isin(list_skills_to_drop)]
    df_problems_valid = df_problems_valid[df_problems_valid['problem_id'].isin(df_skills['problem_id'].unique())]
    df_skills_merged = pd.merge(df_problems_valid, df_skills, on='problem_id')

    user_skill_counts = df_skills_merged.groupby(['skill_id', 'user_id']).size().reset_index(name='user_count')
    df_max_user_count_per_skill = user_skill_counts.groupby('skill_id')['user_count'].max().reset_index()
    df_skill_ids_to_keep = df_max_user_count_per_skill[df_max_user_count_per_skill['user_count'] > min_len_longest_seq]
    list_skills_to_keep = df_skill_ids_to_keep['skill_id'].unique()
    df_skills = df_skills[df_skills['skill_id'].isin(list_skills_to_keep)]
    df_problems_valid = df_problems_valid[df_problems_valid['problem_id'].isin(df_skills['problem_id'].unique())]

    return df_problems_valid, df_skills


def return_assistments_dict_bkt(
        df_answers: pd.DataFrame,
        df_skills: pd.DataFrame,
) -> Dict[str, List[List[Literal[0, 1]]]]:
    """
    Returns the assistments data as a Bayesian Knowledge Tracing (BKT) dictionary.

    This function processes the data, grouping it by skill and user, and ensures 
    that the answers meet the minimum criteria before appending them to the dictionary.
    

    Returns:
    - dict: A dictionary where the keys are skill IDs and the values are lists of answers 
      (each answer is a list of integers 0 or 1) for each student.
    """
    # Load the dataset
    # Initialize a dictionary to map skill_ids to lists of users' answers
    skill_dict = defaultdict(list)

    df_data = pd.merge(df_answers, df_skills, on='problem_id')

    # Group the DataFrame by skill_id and user_id
    for skill_id, skill_group in df_data.groupby('skill_id'):
        answer_list = []
        for _, user_group in skill_group.groupby('user_id'):
            # Extract the list of answers for this user
            answers = user_group['correct'].tolist()
            answer_list.append(answers)
    
        skill_dict[skill_id] = answer_list

    # Convert defaultdict to a regular dictionary and return
    skill_dict = dict(skill_dict)

    return skill_dict


def return_assistments_dict_dkt(
        input_file: str,
        max_sequence_len: int,
        min_appearances_per_problem: int,
        min_answers_per_user: int
) -> Dict[str, List[Tuple[int, Literal[0, 1]]]]:
    """
    Returns the assistments data as a Deep Knowledge Tracing (DKT) dictionary.

    This function processes the data, filtering out problems and users that don't meet 
    the required frequency criteria, and generates a dictionary of answers for each user.
    
    Parameters:
    - input_file: str, path to the input CSV file containing the assistments data.
    - max_sequence_len: int, The maximum sequence length allowed for each user.
    - min_appearances_per_problem: int, The minimum number of appearances required for a problem to be valid.
    - min_answers_per_user: int, The minimum number of answers required from each user.

    Returns:
    - dict: A dictionary where the keys are user IDs and the values are lists of tuples 
      (problem_id, correct) representing each user’s interactions with problems.
    """
    # Load the dataset
    df_data = return_assistments_df_dkt(input_file, max_sequence_len,
                                        min_appearances_per_problem,
                                        min_answers_per_user)

    # Create the desired dictionary where each user_id maps to a list of (problem_id, correct) tuples
    result_dict = defaultdict(list)

    # Group by 'user_id' and iterate through each group
    for user_id, user_group in df_data.groupby('user_id'):
        result_dict[user_id] = list(zip(user_group['problem_id'], user_group['correct']))

    return result_dict

if __name__ == "__main__":
    # Define constants
    INPUT_FILE = 'data/raw/skill_builder_data.csv'
    OUTPUT_FOLDER = 'data/preprocessed/'
    output_file_bkt = 'data/preprocessed/assistments_skill_dict.json'

    output_file_dkt = 'data/preprocessed/assistments_user_dict.json'
    MAX_USER_SEQUENCE_LEN = 400
    MIN_USER_SEQUENCE_LEN = 5
    MIN_ANSWERS_PER_PROBLEM = 10
    MIN_USERS_PER_SKILL = 10
    MIN_LEN_LONGEST_SEQ = 3

    df_answers, df_skills = clean_assistments_data(
        input_file=INPUT_FILE,
        max_user_sequence_len=MAX_USER_SEQUENCE_LEN,
        min_user_sequence_len=MIN_USER_SEQUENCE_LEN,
        min_answers_per_problem=MIN_ANSWERS_PER_PROBLEM,
        min_user_per_skill=MIN_USERS_PER_SKILL,
        min_len_longest_seq=MIN_LEN_LONGEST_SEQ,
        )
    
    num_users = len(df_answers['user_id'].unique())
    num_problems = len(df_skills['problem_id'].unique())
    num_skills = len(df_skills['skill_id'].unique())
    num_answers = len(df_answers)
    print(f'Number of users: {num_users}')
    print(f'Number of problems: {num_problems}')
    print(f'Number of skills: {num_skills}')
    print(f'Number of answers: {num_answers}')

    # Process the assistments data and generate the skill dictionary (for BKT)
    skill_dict = return_assistments_dict_bkt(
        input_file, min_students_per_skill, min_sequence_length_per_skill)

    # Save the skill dictionary to a JSON file
    with open(output_file_bkt, 'w') as json_file:
        json.dump(skill_dict, json_file)
    print(f"Skill dictionary saved to {output_file_bkt}.")

    # Process the assistments data and generate the user dictionary (for DKT)
    user_dict = return_assistments_dict_dkt(
        input_file, max_sequence_len, min_appearances_per_problem, min_answers_per_user)

    # Save the user dictionary to a JSON file
    with open(output_file_dkt, 'w') as json_file:
        json.dump(user_dict, json_file)
    print(f"User dictionary saved to {output_file_dkt}.")
