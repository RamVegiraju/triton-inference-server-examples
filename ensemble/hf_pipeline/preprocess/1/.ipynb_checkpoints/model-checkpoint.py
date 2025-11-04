import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer

MAX_LEN = 128
MODEL_NAME = "distilbert-base-uncased"  # or your tokenizer

class TritonPythonModel:
    def initialize(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def execute(self, requests):
        responses = []
        for request in requests:
            text_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            texts = [t.decode("utf-8") for t in text_tensor.as_numpy().tolist()]

            enc = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="np"
            )

            input_ids = enc["input_ids"].astype("int64")
            attention_mask = enc["attention_mask"].astype("int64")

            out_tensors = [
                pb_utils.Tensor("INPUT_IDS", input_ids),
                pb_utils.Tensor("ATTENTION_MASK", attention_mask),
            ]
            responses.append(pb_utils.InferenceResponse(output_tensors=out_tensors))
        return responses
