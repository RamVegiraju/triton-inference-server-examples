import json
import logging
import numpy as np 
import subprocess
import sys
import triton_python_backend_utils as pb_utils
import transformers
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TritonPythonModel:
    """This model loops through different dtypes to make sure that
    serialize_byte_tensor works correctly in the Python backend.
    """

    def initialize(self, args):
        # initialize tokenizer
        self.model = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
        
    def execute(self, requests):
        responses = []
        for request in requests:
            sampInput = pb_utils.get_input_tensor_by_name(request, "text")
            inpData = sampInput.as_numpy()
            #convert list item to string for model to process
            decoded_str = inpData[0].decode('utf-8')
            sentiment = self.model(decoded_str)
            sent_array = np.array(sentiment)
            out_tensor = pb_utils.Tensor("sent_arr", sent_array)
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses
