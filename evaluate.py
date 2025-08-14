import math


def evaluate_output(model_prediction, expected_output):
    """Parameters:
    model_prediction: The daily revenue predicted by the model
    expected_output: The daily revenue reported in the instance"""
    return 1/math.log(abs(expected_output - model_prediction) + 1)