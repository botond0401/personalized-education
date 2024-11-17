import json
from collections import defaultdict
import pandas as pd
from typing import List

MAX_LEN = 256

def _valid_answers(
        answers: List[List[int]],
        min_students: int,
        min_sequence_length: int
) -> bool:
    max_sequence_length = max([len(answer) for answer in answers])
    if len(answers) >= min_students and max_sequence_length >= min_sequence_length:
        return True
    return False


def return_assistments_dict_bkt(
        input_file,
        min_students = 10,
        min_sequence_length = 3
        ) -> None:
    """
    Save the assistments data as a Bayesian Knowledge Tracing (BKT) dictionary.

    Parameters:
    - input_file: str, path to the input CSV file containing the assistments data.
    - output_file: str, path to the output JSON file where the skill dictionary will be saved.
    """
    
    # Load the dataset
    df_data = return_assistments_df_bkt(input_file)

    # Initialize a dictionary to map skill_ids to lists of users' answers
    skill_dict = defaultdict(list)

    # Group the DataFrame by skill_id and user_id
    for skill_id, skill_group in df_data.groupby('skill_id'):
        for _, user_group in skill_group.groupby('user_id'):
            # Extract the list of answers for this user and append to the skill dictionary
            answers = user_group['correct'].tolist()
            if _valid_answers(answers, min_students, min_sequence_length):
                skill_dict[skill_id].append(answers)

    # Convert defaultdict to a regular dictionary
    skill_dict = dict(skill_dict)

    return skill_dict


def return_assistments_dict_Dkt(
        input_file='../data/raw/skill_builder_data.csv',
        output_file='../data/preprocessed/assistments_user_dict.json',
        constants_file='../data/constants.py'
        ) -> None:
    """
    Save the assistments data as a Deep Knowledge Tracing (BKT) dictionary.

    Parameters:
    - input_file: str, path to the input CSV file containing the assistments data.
    - output_file: str, path to the output JSON file where the skill dictionary will be saved.
    """
    df_data = return_assistments_dkt_df(input_file)

    df_data['problem_id'], _ = pd.factorize(df_data['problem_id'])
    df_data['problem_id'] += 1

    # Extract the highest value
    HIGHEST_PROBLEM_ID = df_data['problem_id'].max()

    # Save it to a Python file
    with open(constants_file, "w") as f:
        f.write(f"HIGHEST_PROBLEM_ID = {HIGHEST_PROBLEM_ID}\n")
        
    # Create the desired dictionary
    result_dict = defaultdict(list)

    # Group by 'user_id' and iterate through each group
    for user_id, user_group in df_data.groupby('user_id'):
        # Create a list of tuples (problem_id, correct) for each user_id
        result_dict[user_id] = list(zip(user_group['problem_id'], user_group['correct']))

    # Save the skill dictionary to a JSON file
    with open(output_file, 'w') as json_file:
        json.dump(result_dict, json_file)  # Added indent for better readability

    print(f"Skill dictionary saved to {output_file}.")


def return_assistments_df_bkt(
        input_file='../data/raw/skill_builder_data.csv'
        ) -> pd.DataFrame:
    """
    Return the assistments data as preprocessed a pandas dataframe for BKT.

    Parameters:
    - input_file: str, path to the input CSV file containing the assistments data.
    """
    
    # Load the dataset
    df_assistments = pd.read_csv(input_file, encoding='ISO-8859-1', low_memory=False)

    # Prepare the data: select relevant columns and drop missing values
    df_data = df_assistments[['order_id', 'user_id', 'correct', 'skill_id', 'problem_id'
                              'ms_first_response', 'bottom_hint']]
    # Drop rows where any of the specified columns have missing values
    df_data = df_data.dropna(subset=['order_id', 'user_id', 'correct', 'skill_id'])
    df_data = df_data.drop_duplicates(subset=['order_id', 'user_id', 'correct', 'skill_id'])
    df_data['order_id'] = df_data['order_id'].astype(int)
    df_data['user_id'] = df_data['user_id'].astype(int)
    df_data['correct'] = df_data['correct'].astype(int)
    df_data['skill_id'] = df_data['skill_id'].astype(int)
    df_data = df_data.sort_values(by=['user_id', 'order_id'])

    return df_data


def return_assistments_dkt_df(
        input_file='../data/raw/skill_builder_data.csv'
        ) -> pd.DataFrame:
    """
    Return the assistments data as a pandas dataframe for DKT.

    Parameters:
    - input_file: str, path to the input CSV file containing the assistments data.
    """
    
    # Load the dataset
    df_assistments = pd.read_csv(input_file, encoding='ISO-8859-1', low_memory=False)

    df_data = df_assistments[['order_id', 'user_id', 'correct', 'problem_id']]
    # Drop rows where any of the specified columns have missing values
    df_data = df_data.dropna()
    df_data = df_data.drop_duplicates()

    problem_counts = df_data['problem_id'].value_counts()
    problems_to_drop = problem_counts[problem_counts >= 5].index
    df_data = df_data[df_data['problem_id'].isin(problems_to_drop)]

    user_problem_counts = df_data.groupby('user_id')['problem_id'].nunique()
    users_to_drop = user_problem_counts[user_problem_counts == 1].index
    df_data = df_data[~df_data['user_id'].isin(users_to_drop)]

    df_data['order_id'] = df_data['order_id'].astype(int)
    df_data['user_id'] = df_data['user_id'].astype(int)
    df_data['correct'] = df_data['correct'].astype(int)
    df_data['problem_id'] = df_data['problem_id'].astype(int)
    df_data = df_data.sort_values(by=['user_id', 'order_id'])
    df_data = df_data.groupby('user_id').head(MAX_LEN).reset_index(drop=True)

    return df_data



input_file = ='../data/raw/skill_builder_data.csv'
output_file = '../data/preprocessed/assistments_skill_dict.json'
skill_dict = convert_assistments_df_to_dict_BKT(input_file)

# Save the skill dictionary to a JSON file
with open(output_file, 'w') as json_file:
    json.dump(skill_dict, json_file)  # Added indent for better readability

print(f"Skill dictionary saved to {output_file}.")