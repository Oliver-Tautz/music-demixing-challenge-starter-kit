
from test_umx import UMXPredictor
from test_xumx import XUMXPredictor
from test import DemucsPredictor



# UMX needs `models` folder to be present in your submission, check test_umx.py to learn more
umx_predictor = UMXPredictor()

# X-UMX needs `models` folder to be present in your submission, check test_xumx.py to learn more
xumx_predictor = XUMXPredictor()

"""
PARTICIPANT_TODO: The implementation you want to submit as your submission
"""
submission = DemucsPredictor(noise_reduction=False,noise_threshold=0.01)
submission.run()
print("Successfully completed music demixing...")

