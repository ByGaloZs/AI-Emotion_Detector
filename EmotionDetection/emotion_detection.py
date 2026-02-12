"""Utilities for detecting emotions in text using an external API.

This module exposes `emotion_detector`, which sends text to the Watson NLP
service, processes the JSON response, and returns emotion scores together
with the dominant emotion.
"""

import json

import requests


def emotion_detector(text_to_analyse):
    """Analyze emotions in input text and return category scores.

    The function builds the payload required by the emotion API,
    performs an HTTP `POST` request, parses the response, and extracts
    the following metrics:
    - `anger`
    - `disgust`
    - `fear`
    - `joy`
    - `sadness`

    It also computes `dominant_emotion` as the emotion with the highest score.

    Args:
        text_to_analyse (str): Text to analyze.

    Returns:
        dict: Dictionary containing emotion scores and the dominant emotion.
            If the API returns status code 400, all keys are returned with
            value `None`.
    """
    # Watson Emotion Predict service endpoint.
    url = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"

    # API payload: raw document containing the input text.
    myobj = {"raw_document": {"text": text_to_analyse}}

    # Required header to select the emotion model.
    header = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}

    # Send request to the remote inference service.
    response = requests.post(url, json=myobj, headers=header)

    # Handle invalid or empty input reported by the API.
    if response.status_code == 400:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

    # Parse JSON response text into a Python dictionary.
    formatted_response = json.loads(response.text)

    # Extract emotions from the first prediction result.
    emotions = formatted_response["emotionPredictions"][0]["emotion"]

    # Assign each score to a dedicated variable for readability.
    anger = emotions["anger"]
    disgust = emotions["disgust"]
    fear = emotions["fear"]
    joy = emotions["joy"]
    sadness = emotions["sadness"]

    # Group scores to determine the dominant emotion.
    scores = {
        "anger": anger,
        "disgust": disgust,
        "fear": fear,
        "joy": joy,
        "sadness": sadness,
    }

    # Select the emotion with the highest score.
    dominant_emotion = max(scores, key=scores.get)

    # Return normalized output for downstream use.
    return {
        "anger": anger,
        "disgust": disgust,
        "fear": fear,
        "joy": joy,
        "sadness": sadness,
        "dominant_emotion": dominant_emotion,
    }
