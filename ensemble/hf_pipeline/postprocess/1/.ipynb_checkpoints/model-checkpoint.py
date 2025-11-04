import triton_python_backend_utils as pb_utils
import numpy as np

LABELS = ["NEGATIVE", "POSITIVE"]

class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            logits_tensor = pb_utils.get_input_tensor_by_name(request, "LOGITS")
            logits = logits_tensor.as_numpy()                  # [B, num_labels]
            preds = logits.argmax(axis=-1)                     # [B]

            out = np.array(
                [LABELS[i].encode("utf-8") for i in preds],
                dtype=object
            )

            out_tensor = pb_utils.Tensor("LABEL", out)
            responses.append(pb_utils.InferenceResponse(
                output_tensors=[out_tensor]
            ))
        return responses
