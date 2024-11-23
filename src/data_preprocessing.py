import json
from collections import defaultdict
from typing import List, Dict, Tuple, Literal

import pandas as pd


def return_assistments_df_bkt(input_file: str) -> pd.DataFrame:
    """
    Returns the assistments data as a preprocessed pandas dataframe for BKT.

    This function loads the assistments data from a CSV file, selects relevant columns,
    drops rows with missing values, removes duplicates, and sorts the data by user and order IDs.
    
    Parameters:
    - input_file: str, path to the input CSV file containing the assistments data.
    
    Returns:
    - pd.DataFrame: A preprocessed dataframe containing the assistments data, 
      with relevant columns, cleaned and sorted.
    """
    # Load the dataset with encoding to handle non-UTF-8 characters
    df_assistments = pd.read_csv(input_file, encoding='ISO-8859-1', low_memory=False)

    # Prepare the data: select relevant columns and drop missing values
    df_data = df_assistments[['order_id', 'user_id', 'correct', 'skill_id', 'problem_id',
                              'ms_first_response', 'bottom_hint', 'opportunity']]
    df_data = df_data.copy()
    # Pad the 'opportunity' column with leading zeros to make it a 4-digit string
    df_data['opportunity_padded'] = df_data.loc[:, 'opportunity'].apply(lambda x: f"{int(x):04d}")

    # Create the 'timestamp' column by concatenating 'order_id' and padded 'opportunity'
    df_data['timestamp'] = df_data.loc[:, 'order_id'].astype(str) + df_data['opportunity_padded']


    # Drop rows where any of the specified columns have missing values
    df_data = df_data.dropna(subset=['timestamp', 'user_id', 'correct', 'skill_id'])
    df_data = df_data.drop_duplicates(subset=['timestamp', 'user_id', 'correct', 'problem_id','skill_id'])

    
    # Convert columns to appropriate data types
    df_data['timestamp'] = df_data['timestamp'].astype(int)
    df_data['user_id'] = df_data['user_id'].astype(int)
    df_data['correct'] = df_data['correct'].astype(int)
    df_data['skill_id'] = df_data['skill_id'].astype(int)
    
    # Sort the dataframe by user ID and order ID to maintain the sequence of events
    df_data = df_data.sort_values(by=['user_id', 'timestamp'])

    return df_data


def return_assistments_df_dkt(
        input_file: str,
        max_sequence_len: int,
        min_appearances_per_problem: int,
        min_answers_per_user: int
        ) -> pd.DataFrame:
    """
    Returns the assistments data as a preprocessed pandas dataframe for DKT.

    This function loads the assistments data from a CSV file, selects relevant columns,
    drops rows with missing values, removes duplicates, and sorts the data by user and order IDs.
    
    Parameters:
    - input_file: str, path to the input CSV file containing the assistments data.
    
    Returns:
    - pd.DataFrame: A preprocessed dataframe containing the assistments data, 
      with relevant columns, cleaned and sorted.
    """
    # Load the dataset with encoding to handle non-UTF-8 characters
    df_assistments = pd.read_csv(input_file, encoding='ISO-8859-1', low_memory=False)

    # Prepare the data: select relevant columns and drop missing values
    df_data = df_assistments[['order_id', 'user_id', 'correct', 'skill_id', 'problem_id',
                              'ms_first_response', 'bottom_hint', 'opportunity']]
    df_data = df_data.copy()
        # Pad the 'opportunity' column with leading zeros to make it a 4-digit string
    df_data['opportunity_padded'] = df_data.loc[:, 'opportunity'].apply(lambda x: f"{int(x):04d}")

    # Create the 'timestamp' column by concatenating 'order_id' and padded 'opportunity'
    df_data['timestamp'] = df_data.loc[:, 'order_id'].astype(str) + df_data['opportunity_padded']

    # Drop rows where any of the specified columns have missing values
    df_data = df_data.dropna(subset=['timestamp', 'user_id', 'correct', 'problem_id'])
    
    # Remove duplicate rows based on the specified subset of columns
    df_data = df_data.drop_duplicates(subset=['timestamp', 'user_id', 'correct', 'problem_id'])

    # Group the data by user_id and limit each user to a maximum sequence length
    df_data = df_data.groupby('user_id').head(max_sequence_len).reset_index(drop=True)

    # Filter problems based on the number of appearances
    problem_counts = df_data['problem_id'].value_counts()
    problems_to_keep = problem_counts[problem_counts >= min_appearances_per_problem].index
    df_data = df_data[df_data['problem_id'].isin(problems_to_keep)]

    # Filter users based on the number of unique problems they've interacted with
    user_problem_counts = df_data.groupby('user_id')['problem_id'].nunique()
    users_to_keep = user_problem_counts[user_problem_counts >= min_answers_per_user].index
    df_data = df_data[df_data['user_id'].isin(users_to_keep)]
    
    # Convert columns to appropriate data types
    df_data['timestamp'] = df_data['timestamp'].astype(int)
    df_data['user_id'] = df_data['user_id'].astype(int)
    df_data['correct'] = df_data['correct'].astype(int)
    df_data['problem_id'] = df_data['problem_id'].astype(int)
    
    # Sort the dataframe by user ID and order ID to maintain the sequence of events
    df_data = df_data.sort_values(by=['user_id', 'timestamp'])

    return df_data


def _valid_answers(
        answers: List[List[int]],
        min_students_per_skill: int,
        min_sequence_length_per_skill: int
) -> bool:
    """
    Validates if the given answers meet the minimum required conditions.

    This function checks if the skill has a sufficient number of students and if the 
    maximum sequence length of answers per student exceeds the minimum threshold.

    Parameters:
    - answers: List[List[int]], A list of answers for different users (each user has a list of answers).
    - min_students_per_skill: int, The minimum number of students required for a skill to be valid.
    - min_sequence_length_per_skill: int, The minimum sequence length required for a skill to be valid.

    Returns:
    - bool: True if the skill is valid based on the criteria, otherwise False.
    """
    # Calculate the maximum sequence length across all users' answers
    max_sequence_length = max(len(answer) for answer in answers)
    
    # Check if the number of students and the max sequence length meet the minimum requirements
    if len(answers) >= min_students_per_skill and max_sequence_length >= min_sequence_length_per_skill:
        return True
    return False


def return_assistments_dict_bkt(
        input_file: str,
        min_students_per_skill: int,
        min_sequence_length_per_skill: int
) -> Dict[str, List[List[Literal[0, 1]]]]:
    """
    Returns the assistments data as a Bayesian Knowledge Tracing (BKT) dictionary.

    This function processes the data, grouping it by skill and user, and ensures 
    that the answers meet the minimum criteria before appending them to the dictionary.
    
    Parameters:
    - input_file: str, path to the input CSV file containing the assistments data.
    - min_students_per_skill: int, The minimum number of students required per skill.
    - min_sequence_length_per_skill: int, The minimum sequence length required per skill.

    Returns:
    - dict: A dictionary where the keys are skill IDs and the values are lists of answers 
      (each answer is a list of integers 0 or 1) for each student.
    """
    # Load the dataset
    df_data = return_assistments_df_bkt(input_file)
    
    # Initialize a dictionary to map skill_ids to lists of users' answers
    skill_dict = defaultdict(list)

    # Group the DataFrame by skill_id and user_id
    for skill_id, skill_group in df_data.groupby('skill_id'):
        answer_list = []
        for _, user_group in skill_group.groupby('user_id'):
            # Extract the list of answers for this user
            answers = user_group['correct'].tolist()
            answer_list.append(answers)
    
        # Validate the answers and add them to the dictionary if valid
        if _valid_answers(answer_list, min_students_per_skill, min_sequence_length_per_skill):
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
    """
    Main function to preprocess assistments data and save the generated dictionaries as JSON files.

    This script loads raw assistments data, processes it for Bayesian Knowledge Tracing (BKT) 
    and Deep Knowledge Tracing (DKT), and saves the results into JSON files.
    """
    # Define constants
    input_file = 'data/raw/skill_builder_data.csv'
    output_file_bkt = 'data/preprocessed/assistments_skill_dict.json'
    min_students_per_skill = 6
    min_sequence_length_per_skill = 3

    output_file_dkt = 'data/preprocessed/assistments_user_dict.json'
    max_sequence_len = 256
    min_appearances_per_problem = 5
    min_answers_per_user = 3

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
