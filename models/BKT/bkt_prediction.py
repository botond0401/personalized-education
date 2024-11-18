"""
bkt_prediction.py

This module provides functionality for making predictions using a 
Bayesian Knowledge Tracing (BKT) model. It includes methods for predicting 
the next hidden state distribution, next observation distribution, 
and the next observation value based on a sequence of user answers.

Classes:
- BKTPrediction: Contains methods for predicting user performance 
  based on the BKT model.
"""


class BKTPrediction:
    """
    BKTPrediction Class

    A utility class for predicting the next hidden states and observations 
    based on a given sequence of user answers using a BKT model. This class 
    provides methods for estimating future states and observations.

    Methods:
    - predict_next_hidden_state_distribution(model, observations): 
      Predicts the probability distribution over hidden states for the next step.
    - predict_next_observation_distribution(model, observations): 
      Predicts the probability distribution over the next observation value.
    - predict_next_observation(model, observations): 
      Predicts the next observation value based on previous observations.
    - predict(model, observations): 
      Predicts the next observation values for a sequence of observations.
    """

    @staticmethod
    def predict_next_hidden_state_distribution(model, observations):
        """
        Predict the probability distribution over hidden states for the next step.

        Parameters:
        - model: An instance of the ModelBKT class.
        - observations: A sequence of observations (e.g., user answers).

        Returns:
        - next_state_probs: A probability distribution over the hidden states for the next time step.
        """
        # Initialize forward probabilities with the starting probabilities
        forward_probs = model.startprob_

        # Iterate through each observation in the sequence
        for obs in observations:
            obs_value = obs[0] if isinstance(obs, list) else obs
            # Calculate forward probabilities for the next time step
            forward_probs = forward_probs @ model.transmat_ * model.emissionprob_[:, obs_value]

            # Normalize the forward probabilities
            forward_probs /= forward_probs.sum()

        return forward_probs

    @staticmethod
    def predict_next_observation_distribution(model, observations):
        """
        Predict the probability distribution over the next observation value.

        Parameters:
        - model: An instance of the ModelBKT class.
        - observations: A sequence of observations (e.g., user answers).

        Returns:
        - next_obs_probs: A probability distribution over the possible observation values (0 or 1).
        """
        # First, predict the current hidden state distribution
        current_hidden_state_probs = BKTPrediction.predict_next_hidden_state_distribution(model, observations)

        # Compute the next observation probabilities based on current hidden state probabilities
        next_obs_probs = current_hidden_state_probs @ model.emissionprob_

        # Normalize the next observation probabilities
        next_obs_probs /= next_obs_probs.sum()

        return next_obs_probs

    @staticmethod
    def predict_next_observation(model, observations):
        """
        Predict the next observation value based on previous observations.

        Parameters:
        - model: An instance of the ModelBKT class.
        - observations: A sequence of observations (e.g., user answers).

        Returns:
        - prediction: The predicted next observation value (0 or 1).
        """
        next_obs_probs = BKTPrediction.predict_next_observation_distribution(model, observations)
        return float(next_obs_probs[1])  # Probability of observing 1

    @staticmethod
    def predict(model, observations):
        """
        Predict the next observation values for a sequence of observations.

        Parameters:
        - model: An instance of the ModelBKT class.
        - observations: A sequence of observations (e.g., user answers) or a list of sequences.

        Returns:
        - predictions: A list of predicted next observation values.
        """
        predictions = []
        if isinstance(observations[0], list):
            for sequence in observations:
                for i in range(len(sequence)):
                    current_observations = sequence[:i]
                    prediction = BKTPrediction.predict_next_observation(model, current_observations)
                    predictions.append(prediction)
        else:
            for i in range(len(observations)):
                current_observations = observations[:i]
                prediction = BKTPrediction.predict_next_observation(model, current_observations)
                predictions.append(prediction)

        return predictions
