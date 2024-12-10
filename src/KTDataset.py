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

from collections import defaultdict
from typing import Union, List

import pandas as pd


class KTDataset():
    def __init__(self,
                 df_answers,
                 df_skill_names,
                 renumber_skill_ids=True,
                 prepare_BKT=False,
                 prepare_DKT=False):
        
        self.df_answers = df_answers
        self.df_skill_names = df_skill_names
        self.num_skills = len(df_skill_names)
        self.BKT_datadict = None
        self.DKT_datadict = None


        if renumber_skill_ids:
            # Map original skill IDs to a continuous range starting from 1
            unique_skill_ids = sorted(df_skill_names['skill_id'].astype(int).unique())
            self.skill_id_mapping = {original_id: new_id for new_id, original_id in enumerate(unique_skill_ids, start=1)}
            # Apply the mapping to create the new column
            df_skill_names['skill_ids_renumbered'] = df_skill_names['skill_id'].map(self.skill_id_mapping)
        else:
            self.skill_id_mapping = None

        if prepare_BKT:
            self.create_BKT_datadict()

        if prepare_DKT:
            self.create_DKT_datadict()



    def return_ordered_ids(self, ids: Union[int, str, List[int], List[str]]) -> Union[int, List[int]]:
        """
        Returns renumbered skill IDs based on the mapping.

        Args:
            ids (Union[int, str, List[int], List[str]]): Original skill ID(s) to be renumbered.

        Returns:
            Union[int, List[int]]: Renumbered skill ID(s).
        """

        if isinstance(ids, (int, str)):
            return self.skill_id_mapping[int(ids)]
        elif isinstance(ids, list):
            return [self.skill_id_mapping[int(id)] for id in ids]
        else:
            raise TypeError("IDs must be an int, str, or a list of int/str.")
        

    def return_original_ids(self, ids: Union[int, str, List[int], List[str]]) -> Union[int, List[int]]:
        """
        Returns original skill IDs based on the renumbered skill IDs.

        Args:
            ids (Union[int, str, List[int], List[str]]): Renumbered skill ID(s) to be mapped back to the original IDs.

        Returns:
            Union[int, str, List[int], List[str]]: Original skill ID(s).
        """

        if isinstance(ids, (int, str)):
            return self.original_id_mapping.get(int(ids), None)  # Convert to integer for lookup
        elif isinstance(ids, list):
            return [self.original_id_mapping.get(int(id), None) for id in ids]  # Convert each to integer for lookup
        else:
            raise TypeError("IDs must be an int, str, or a list of int/str.")
        

    def create_BKT_datadict(self):
        """
        Converts Assistments data into a Bayesian Knowledge Tracing (BKT) dictionary.

        Groups data by skill ID and user ID, creating a structure where each skill maps 
        to lists of user answer sequences.
        """

        # Create a defaultdict to collect answer sequences per skill
        skill_dict = defaultdict(list)

        for _, row in self.df_skill_names.iterrows():
            skill_id = row['skill_id']
            skill_name = row['skill_name']

            df_answers_filtered = self.df_answers[self.df_answers[str(skill_id)] == 1]
            # Group by user_id and collect sequences of correct answers
            answer_list = df_answers_filtered.groupby('user_id')['correct'].apply(list)

            # Collect all answer sequences for the current skill
            skill_dict[f"{skill_id} ({skill_name})"] = answer_list.tolist()

        # Convert defaultdict to a regular dictionary and return
        self.BKT_datadict = dict(skill_dict)


    def create_DKT_datadict(self, additional_columns=None):
        """
        Create a DKT dataset with one-hot vectors and additional columns if specified.
        
        Args:
            additional_columns (list, optional): List of column names to include in the tuples, 
                                                e.g., ['bottom_hint', 'ms_first_response']. Defaults to None.
        """
        # Dictionary to hold the data for each user
        dict_skills = defaultdict(list)

        # Iterate over each user group
        for user_id, user_group in self.df_answers.groupby('user_id'):
            # Create a list of tuples for each user's answers
            answer_list = []
            for _, row in user_group.iterrows():
                one_hot_vector = row.iloc[-self.num_skills:].values  # Extract the one-hot vector as an array
                
                # Collect additional column values as a list
                additional_info = row[additional_columns].values if additional_columns else []
                
                is_correct = row['correct']  # Extract if the answer was correct

                # Append the tuple (one-hot vector, additional info, correct flag)
                if additional_columns:
                    answer_list.append((one_hot_vector, additional_info, is_correct))
                else:
                    answer_list.append((one_hot_vector, is_correct))

            # Store the answer list in the dictionary under the user ID
            dict_skills[user_id] = answer_list

        self.DKT_datadict = dict(dict_skills)
