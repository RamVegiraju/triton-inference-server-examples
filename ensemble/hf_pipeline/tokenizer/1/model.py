import json
import logging
import numpy as np 
import subprocess
import sys

import triton_python_backend_utils as pb_utils
import transformers

#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TritonPythonModel:
    """This model loops through different dtypes to make sure that
    serialize_byte_tensor works correctly in the Python backend.
    """

    def initialize(self, args):
        # initialize tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("sentence-transformers/msmarco-bert-base-dot-v5")
        
    def execute(self, requests):
        
        responses = []
        for request in requests:
            sampInput = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            inpData = sampInput.as_numpy()
            
            #convert to string to tokenize
            decoded_str = inpData[0].decode('utf-8')
            tokenized_tensors = self.tokenizer(decoded_str, padding=True, truncation=True, return_tensors='pt')
            
            input_ids = np.array(tokenized_tensors['input_ids'])
            token_type_ids = np.array(tokenized_tensors['token_type_ids'])
            attention_mask = np.array(tokenized_tensors['attention_mask'])
            
            out_tensor_0 = pb_utils.Tensor("input_ids", input_ids)
            out_tensor_1 = pb_utils.Tensor("token_type_ids", token_type_ids)
            out_tensor_2 = pb_utils.Tensor("attention_mask", attention_mask)

            responses.append(pb_utils.InferenceResponse([out_tensor_0,out_tensor_1, out_tensor_2]))
        return responses
