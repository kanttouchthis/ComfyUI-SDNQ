from comfy.model_management import unload_all_models

from .loader import SDNQLoader, SDNQLoraLoader
NODE_CLASS_MAPPINGS = {"SDNQLoader":SDNQLoader, "SDNQLoraLoader":SDNQLoraLoader}