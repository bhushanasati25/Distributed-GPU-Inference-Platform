# =============================================================================
# Triton Python Backend: Image Preprocessor
# =============================================================================
import io
import json
import numpy as np
from PIL import Image

try:
    import triton_python_backend_utils as pb_utils
except ImportError:
    pb_utils = None


class TritonPythonModel:
    """Processes raw incoming JPEG/PNG byte streams into normalized NCHW tensors."""

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        self.output_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(self.model_config, "preprocessed_tensor")["data_type"]
        ) if pb_utils else np.float32

        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((3, 1, 1))
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((3, 1, 1))

    def execute(self, requests):
        responses = []
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "raw_image_bytes")
            byte_array = in_tensor.as_numpy()

            batch_processed = []
            for raw_bytes in byte_array:
                # Decode bytes with PIL
                img = Image.open(io.BytesIO(raw_bytes[0])).convert("RGB")
                img = img.resize((224, 224), Image.Resampling.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0  # HWC

                # HWC -> CHW
                arr = np.transpose(arr, (2, 0, 1))
                # Normalize
                arr = (arr - self.mean) / self.std
                batch_processed.append(arr)

            out_np = np.stack(batch_processed, axis=0).astype(self.output_dtype)
            out_tensor = pb_utils.Tensor("preprocessed_tensor", out_np)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses

    def finalize(self):
        pass

