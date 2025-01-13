"""
bkt_initialization.py

This module provides functionality for initializing the parameters of a 
Bayesian Knowledge Tracing (BKT) model using user answer data. It includes 
methods for calculating initial probabilities, transition probabilities, 
and emission probabilities based on observed user performance.

Classes:
- BKTInitialization: Contains methods for setting initial model parameters.
"""

import numpy as np


class BKTInitialization:
    """
    BKTInitialization Class

    A utility class for calculating and setting the initial parameters of 
    a Bayesian Knowledge Tracing (BKT) model. This class encapsulates 
    methods for determining initial, transition, and emission probabilities 
    based on sequences of user answers.

    Methods:
    - set_initial_parameters(model, allow_forget=False): Calculates and sets
      the initial, transition, and emission probabilities for the provided
      model.
    - _calculate_answer_counts(model): Computes counts of user answers
      for transition probability calculations.
    - _update_answer_counts(answer_counts, previous_answer, current_answer):
      Updates the answer counts based on previous and current answers.
    - _calculate_initial_probabilities(correct_first_answer_count,
    total_users):
      Calculates initial probabilities for the BKT model.
    - _calculate_transition_probabilities(answer_counts, allow_forget):
      Computes the transition probabilities for the model.
    - _calculate_emission_probabilities(answer_counts): Computes the 
      emission probabilities for the model.
    """

    @staticmethod
    def set_initial_parameters(model, allow_forget=False):
        """
        Automatically calculates and sets initial, transition, and emission
        probabilities based on user answer patterns.

        Parameters:
        - model: An instance of the ModelBKT class.
        - allow_forget: Boolean flag indicating whether forgetting is allowed
        (default: False).
        """
        answer_counts, correct_first_answer_count = (
            BKTInitialization._calculate_answer_counts(model)
            )

        # Set initial probabilities
        model.startprob_ = (
            BKTInitialization._calculate_initial_probabilities(
                correct_first_answer_count, model.n_users
                )
            )

        # Calculate transition and emission probabilities
        transition_probabilities = (
            BKTInitialization._calculate_transition_probabilities(
                answer_counts, allow_forget
                )
            )
        emission_probabilities = (
            BKTInitialization._calculate_emission_probabilities(answer_counts)
            )

        model.transmat_ = transition_probabilities
        model.emissionprob_ = emission_probabilities

    @staticmethod
    def _calculate_answer_counts(model):
        """
        Calculate answer counts and the number of correct first answers.

        Parameters:
        - model: An instance of the ModelBKT class.

        Returns:
        - answer_counts: A dictionary of answer counts.
        - correct_first_answer_count: The number of users who answered
        correctly first.
        """
        answer_counts = {('No', 'No'): 0,
                         ('No', 'Yes'): 0,
                         ('Yes', 'No'): 0,
                         ('Yes', 'Yes'): 0}
        correct_first_answer_count = 0

        for user_answers in model.user_answers:
            if user_answers[0] == 1:
                correct_first_answer_count += 1
            previous_answer = 0  # Assume starting state

            for answer in user_answers:
                BKTInitialization._update_answer_counts(
                    answer_counts, previous_answer, answer
                    )
                previous_answer = answer

        return answer_counts, correct_first_answer_count

    @staticmethod
    def _update_answer_counts(answer_counts, previous_answer, current_answer):
        """
        Update the answer counts based on previous and current answers.

        Parameters:
        - answer_counts: A dictionary of answer counts.
        - previous_answer: The previous answer (0 or 1).
        - current_answer: The current answer (0 or 1).
        """
        if previous_answer == 0 and current_answer == 0:
            answer_counts[('No', 'No')] += 1
        elif previous_answer == 0 and current_answer == 1:
            answer_counts[('No', 'Yes')] += 1
        elif previous_answer == 1 and current_answer == 0:
            answer_counts[('Yes', 'No')] += 1
        elif previous_answer == 1 and current_answer == 1:
            answer_counts[('Yes', 'Yes')] += 1

    @staticmethod
    def _calculate_initial_probabilities(
        correct_first_answer_count,
        total_users
        ):
        """
        Calculate the initial probabilities.

        Parameters:
        - correct_first_answer_count: The number of users who answered
        correctly first.
        - total_users: The total number of users.

        Returns:
        - A numpy array representing the initial probabilities.
        """
        incorrect_first_answer_count = total_users - correct_first_answer_count
        return np.array([incorrect_first_answer_count / total_users, 
                         correct_first_answer_count / total_users])

    @staticmethod
    def _calculate_transition_probabilities(answer_counts, allow_forget):
        """
        Calculate transition probabilities.

        Parameters:
        - answer_counts: A dictionary of answer counts.
        - allow_forget: Boolean flag indicating whether forgetting is allowed.

        Returns:
        - A numpy array representing the transition probabilities.
        """
        no_to_yes_count = answer_counts[('No', 'Yes')] / 2 # because of guesses
        no_to_no_count = answer_counts[('No', 'No')] + no_to_yes_count
        total_no_count = answer_counts[('No', 'Yes')] + answer_counts[('No', 'No')]

        if allow_forget:
            yes_to_no_count = answer_counts[('Yes', 'No')] / 2 # becuase of mistakes
            yes_to_yes_count = answer_counts[('Yes', 'Yes')] + yes_to_no_count
            total_yes_count = yes_to_no_count + yes_to_yes_count
            return np.array([[no_to_no_count / total_no_count if total_no_count > 0 else 0.5,
                               no_to_yes_count / total_no_count if total_no_count > 0 else 0.5],
                              [yes_to_no_count / total_yes_count if total_yes_count > 0 else 0.5,
                              yes_to_yes_count / total_yes_count if total_yes_count > 0 else 0.5]])
        return np.array([[no_to_no_count / total_no_count if total_no_count > 0 else 0.5,
                            no_to_yes_count / total_no_count if total_no_count > 0 else 0.5],
                            [0, 1]])

    @staticmethod
    def _calculate_emission_probabilities(answer_counts):
        """
        Calculate emission probabilities.

        Parameters:
        - answer_counts: A dictionary of answer counts.

        Returns:
        - A numpy array representing the emission probabilities.
        """
        no_to_yes_count = answer_counts[('No', 'Yes')] / 2
        no_to_no_count = answer_counts[('No', 'No')] + no_to_yes_count
        total_no_count = answer_counts[('No', 'Yes')] + answer_counts[('No', 'No')]
        yes_to_no_count = answer_counts[('Yes', 'No')]
        yes_to_yes_count = answer_counts[('Yes', 'Yes')]
        total_yes_count = yes_to_no_count + yes_to_yes_count

        return np.array([[no_to_no_count / total_no_count if total_no_count > 0 else 0.5,
                           no_to_yes_count / total_no_count if total_no_count > 0 else 0.5],
                          [yes_to_no_count / total_yes_count if total_yes_count > 0 else 0.5,
                           yes_to_yes_count / total_yes_count if total_yes_count > 0 else 0.5]])
