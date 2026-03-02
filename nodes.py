import os
import torch
import numpy as np
from PIL import Image

import folder_paths

# Scan all registered checkpoint directories for HuggingFace model subdirs
# (identified by the presence of config.json). This picks up models on any
# drive mapped in extra_model_paths.yaml as well as the default models/ folder.
def _list_model_dirs():
    checkpoint_dirs = folder_paths.get_folder_paths("checkpoints")
    found = {}  # label -> absolute path

    for base in checkpoint_dirs:
        if not os.path.isdir(base):
            continue
        for entry in os.listdir(base):
            full = os.path.join(base, entry)
            if os.path.isdir(full) and os.path.isfile(os.path.join(full, "config.json")):
                found[entry] = full  # later paths override earlier if same name

    return (list(found.keys()), found) if found else (
        ["(no HuggingFace model folders found in checkpoints/)"], {}
    )


# Cache at import time so INPUT_TYPES and load_model share the same map
_MODEL_LABELS, _MODEL_PATHS = _list_model_dirs()


class GLMOCRModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_MODEL_LABELS,),
            }
        }

    RETURN_TYPES = ("GLM_OCR_MODEL",)
    RETURN_NAMES = ("glm_ocr_model",)
    FUNCTION = "load_model"
    CATEGORY = "GLM-OCR"

    def load_model(self, model_name):
        import transformers
        from transformers import AutoProcessor, AutoModelForImageTextToText

        model_path = _MODEL_PATHS.get(model_name, model_name)
        print(f"[GLM-OCR] Loading model from: {model_path}")

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        except ValueError as e:
            if "glm_ocr" in str(e) or "does not recognize this architecture" in str(e):
                raise RuntimeError(
                    f"[GLM-OCR] transformers {transformers.__version__} does not yet include "
                    f"the glm_ocr architecture. Fix: run the following command and restart ComfyUI:\n\n"
                    f'  "C:\\ComfyUI_files\\.venv\\Scripts\\pip" install '
                    f"git+https://github.com/huggingface/transformers.git"
                ) from e
            raise

        model.eval()
        print(f"[GLM-OCR] Model loaded successfully.")
        return ({"model": model, "processor": processor},)


class GLMOCRRun:
    TASK_PROMPTS = [
        "Text Recognition:",
        "Formula Recognition:",
        "Table Recognition:",
        "Describe the image.",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glm_ocr_model": ("GLM_OCR_MODEL",),
                "image": ("IMAGE",),
                "task_prompt": (cls.TASK_PROMPTS,),
                "max_new_tokens": ("INT", {
                    "default": 2048, "min": 64, "max": 8192, "step": 64
                }),
            },
            "optional": {
                "custom_prompt": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Leave empty to use task_prompt above",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "GLM-OCR"
    OUTPUT_NODE = True

    def run(self, glm_ocr_model, image, task_prompt, max_new_tokens, custom_prompt=""):
        model = glm_ocr_model["model"]
        processor = glm_ocr_model["processor"]

        prompt = custom_prompt.strip() if custom_prompt.strip() else task_prompt

        # Convert ComfyUI IMAGE tensor [B, H, W, C] → PIL
        img_np = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        output_text = processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        print(f"[GLM-OCR] Output:\n{output_text}")

        return {"ui": {"text": [output_text]}, "result": (output_text,)}


NODE_CLASS_MAPPINGS = {
    "GLMOCRModelLoader": GLMOCRModelLoader,
    "GLMOCRRun": GLMOCRRun,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GLMOCRModelLoader": "GLM-OCR Model Loader",
    "GLMOCRRun": "GLM-OCR Run",
}
