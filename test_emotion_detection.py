import unittest

from EmotionDetection import emotion_detector


class TestEmotionDetector(unittest.TestCase):
    def test_emotion_detector(self):

        # Test case for joy emotion
        test1 = emotion_detector("I am glad this happened")
        self.assertEqual(test1["dominant_emotion"], "joy")

        # Test case for anger emotion
        test2 = emotion_detector("I am really mad about this")
        self.assertEqual(test2["dominant_emotion"], "anger")

        # Test case for disgust emotion
        test3 = emotion_detector("I feel disgusted just hearing about this")
        self.assertEqual(test3["dominant_emotion"], "disgust")

        # Test case for sadness emotion
        test4 = emotion_detector("I am so sad about this")
        self.assertEqual(test4["dominant_emotion"], "sadness")

        # Test case for fear emotion
        test5 = emotion_detector("I am really afraid that this will happen")
        self.assertEqual(test5["dominant_emotion"], "fear")


unittest.main()
