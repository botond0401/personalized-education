import numpy as np
from hmmlearn import hmm
from bkt_initialization import BKTInitialization
from bkt_prediction import BKTPrediction
from bkt_evaluation import calculate_auc

class ModelBKT(hmm.CategoricalHMM):
    def __init__(self, skill_id, user_answers, initial_probs=None, trans_probs=None, emit_probs=None):
        """
        Initializes a BKT model for a specific skill using HMM.

        Parameters:
        - skill_id: The ID of the skill for which the BKT model is created.
        - user_answers: List of sequences of user answers.
        - initial_probs: Initial probabilities for the hidden states (optional).
        - trans_probs: Transition probabilities between states (optional).
        - emit_probs: Emission probabilities for the observed values (optional).
        """
        # Input validation
        if not isinstance(user_answers, list) or not all(isinstance(seq, list) for seq in user_answers):
            raise ValueError("user_answers must be a list of sequences (list of lists).")
        if any(not all(isinstance(ans, int) and ans in [0, 1] for ans in seq) for seq in user_answers):
            raise ValueError("Each user answer must be an integer (0 or 1).")

        # Initialize as a CategoricalHMM with 2 states
        super().__init__(n_components=2, init_params="")

        self.skill_id = skill_id
        self.user_answers = user_answers
        self.n_users = len(user_answers)

        # Set initial parameters if provided
        self.startprob_ = initial_probs
        self.transmat_ = trans_probs
        self.emissionprob_ = emit_probs

    def fit(self):
        """
        Fit the BKT model using user answers.
        """
        if (self.startprob_ is None) or (self.transmat_ is None) or (self.emissionprob_ is None):
            BKTInitialization.set_initial_parameters(self)

        # Prepare the data
        X = np.concatenate([[[answer] for answer in answers] for answers in self.user_answers])
        lengths = [len(answers) for answers in self.user_answers]

        # Fit the HMM model
        super().fit(X, lengths)

    def get_params(self):
        """
        Retrieve the learned parameters of the HMM.

        Returns:
        - A dictionary containing initial, transition, and emission probabilities.
        """
        return {
            'initial_probs': self.startprob_,
            'trans_probs': self.transmat_,
            'emit_probs': self.emissionprob_
        }

    def predict(self, observations):
        """
        Predict the next observation values for a sequence of observations.

        Parameters:
        - observations: A sequence of observations (e.g., user answers) or a list of sequences.

        Returns:
        - predictions: A list of predicted next observation values.
        """
        predictor = BKTPrediction()
        return predictor.predict(self, observations)

    def self_evaluate(self):
        """
        Evaluate the model using AUC score based on user answers.
        """
        predictions = self.predict(self.user_answers)
        return calculate_auc(self.user_answers, predictions)
