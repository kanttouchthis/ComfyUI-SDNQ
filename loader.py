import logging
import os

import comfy.utils
import comfy.lora
from comfy import model_detection, model_management
from comfy.model_management import unload_all_models
from comfy.model_patcher import string_to_seed, get_key_weight
import folder_paths

import torch
import collections
from hqqsvd.linear import HQQSVDLinear, ComfyLora

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def replace_linear(model, new_sd, compute_dtype, int8_matmul, prefix=""):
    for name, module in model.named_children():
        base = f"{prefix}.{name}"
        if (base + ".svd_up") in new_sd:
            W_q = new_sd.pop(f"{prefix}.{name}.weight").to("cuda")
            svd_up = new_sd.pop(f"{prefix}.{name}.svd_up").to("cuda", compute_dtype)
            svd_down = new_sd.pop(f"{prefix}.{name}.svd_down").to("cuda", compute_dtype)
            zero_point = new_sd.pop(f"{prefix}.{name}.zero_point").to("cuda", compute_dtype)
            bias = new_sd.pop(f"{prefix}.{name}.bias").to("cuda", compute_dtype)
            scale = new_sd.pop(f"{prefix}.{name}.scale").to("cuda", compute_dtype)
            nbits = new_sd.pop(f"{prefix}.{name}.nbits", torch.tensor([4]))
            # remove legacy shape
            if new_sd.pop(f"{prefix}.{name}.shape", None) is not None:
                pass

            sdnq_linear = HQQSVDLinear(W_q, svd_up, svd_down, scale, zero_point, bias, nbits.item(), int8_matmul=int8_matmul)
            setattr(model, name, sdnq_linear)
            torch.cuda.empty_cache()
        else:
            if name == "diffusion_model":
                sub_prefix = ""
            elif prefix == "":
                sub_prefix = name
            else:
                sub_prefix = prefix + "." + name
            replace_linear(module, new_sd, compute_dtype, int8_matmul, sub_prefix)
    return model


class CustomPatcher(comfy.model_patcher.ModelPatcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if key not in self.patches:
            return

        weight, set_func, convert_func = get_key_weight(self.model, key)
        inplace_update = self.weight_inplace_update or inplace_update

        if weight.dtype == torch.uint8:
            sdnq_linear = self.model.get_submodule(key.rstrip(".weight"))
            for i, p in enumerate(self.patches[key]):
                k = f"{key}|{i}"
                lora = ComfyLora(k, p, comfy.lora.calculate_weight)
                sdnq_linear.add_lora(lora)
            return

        if key not in self.backup:
            self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(weight.to(device=self.offload_device, copy=inplace_update), inplace_update)

        temp_dtype = comfy.model_management.lora_compute_dtype(device_to)
        if device_to is not None:
            temp_weight = comfy.model_management.cast_to_device(weight, device_to, temp_dtype, copy=True)
        else:
            temp_weight = weight.to(temp_dtype, copy=True)
        if convert_func is not None:
            temp_weight = convert_func(temp_weight, inplace=True)

        out_weight = comfy.lora.calculate_weight(self.patches[key], temp_weight, key)
        if set_func is None:
            out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype, seed=string_to_seed(key))
            if inplace_update:
                comfy.utils.copy_to_param(self.model, key, out_weight)
            else:
                comfy.utils.set_attr_param(self.model, key, out_weight)
        else:
            set_func(out_weight, inplace_update=inplace_update, seed=string_to_seed(key))

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for module in self.model.modules():
                if isinstance(module, HQQSVDLinear):
                    module.loras = {}
        return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)

def load_diffusion_model_state_dict(sd, int8_matmul, model_options={}, metadata=None):
    """
    Loads a UNet diffusion model from a state dictionary, supporting both diffusers and regular formats.

    Args:
        sd (dict): State dictionary containing model weights and configuration
        model_options (dict, optional): Additional options for model loading. Supports:
            - dtype: Override model data type
            - custom_operations: Custom model operations
            - fp8_optimizations: Enable FP8 optimizations

    Returns:
        ModelPatcher: A wrapped model instance that handles device management and weight loading.
        Returns None if the model configuration cannot be detected.

    The function:
    1. Detects and handles different model formats (regular, diffusers, mmdit)
    2. Configures model dtype based on parameters and device capabilities
    3. Handles weight conversion and device placement
    4. Manages model optimization settings
    5. Loads weights and returns a device-managed model instance
    """
    dtype = model_options.get("dtype", None)

    # Allow loading unets from checkpoint files
    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
    temp_sd = comfy.utils.state_dict_prefix_replace(
        sd, {diffusion_model_prefix: ""}, filter_keys=True
    )
    if len(temp_sd) > 0:
        sd = temp_sd

    custom_operations = model_options.get("custom_operations", None)
    if custom_operations is None:
        sd, metadata = comfy.utils.convert_old_quants(sd, "", metadata=metadata)
    parameters = comfy.utils.calculate_parameters(sd)
    weight_dtype = comfy.utils.weight_dtype(sd)

    load_device = model_management.get_torch_device()
    model_config = model_detection.model_config_from_unet(sd, "", metadata=metadata)

    if model_config is not None:
        new_sd = sd
    else:
        new_sd = model_detection.convert_diffusers_mmdit(sd, "")
        if new_sd is not None:  # diffusers mmdit
            model_config = model_detection.model_config_from_unet(new_sd, "")
            if model_config is None:
                return None
        else:  # diffusers unet
            model_config = model_detection.model_config_from_diffusers_unet(sd)
            if model_config is None:
                return None

            diffusers_keys = comfy.utils.unet_to_diffusers(model_config.unet_config)

            new_sd = {}
            for k in diffusers_keys:
                if k in sd:
                    new_sd[diffusers_keys[k]] = sd.pop(k)
                else:
                    logging.warning("{} {}".format(diffusers_keys[k], k))

    offload_device = model_management.unet_offload_device()
    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if model_config.quant_config is not None:
        weight_dtype = None

    if dtype is None:
        unet_dtype = model_management.unet_dtype(
            model_params=parameters,
            supported_dtypes=unet_weight_dtype,
            weight_dtype=weight_dtype,
        )
    else:
        unet_dtype = dtype

    if model_config.quant_config is not None:
        manual_cast_dtype = model_management.unet_manual_cast(
            None, load_device, model_config.supported_inference_dtypes
        )
    else:
        manual_cast_dtype = model_management.unet_manual_cast(
            unet_dtype, load_device, model_config.supported_inference_dtypes
        )
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    if custom_operations is not None:
        model_config.custom_operations = custom_operations

    if model_options.get("fp8_optimizations", False):
        model_config.optimizations["fp8"] = True

    model = model_config.get_model(new_sd, "")
    model = replace_linear(model, new_sd, unet_dtype, int8_matmul)
    model = model.to(offload_device)
    model.load_model_weights(new_sd, "")
    left_over = sd.keys()
    if len(left_over) > 0:
        logging.info("left over keys in diffusion model: {}".format(left_over))
    return CustomPatcher(
        model, load_device=load_device, offload_device=offload_device
    )


def model_detection_error_hint(path, state_dict):
    filename = os.path.basename(path)
    if "lora" in filename.lower():
        return "\nHINT: This seems to be a Lora file and Lora files should be put in the lora folder and loaded with a lora loader node.."
    return ""


def load_diffusion_model(unet_path, int8_matmul, model_options={}):
    sd, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)
    model = load_diffusion_model_state_dict(
        sd, int8_matmul, model_options=model_options, metadata=metadata
    )
    if model is None:
        logging.error("ERROR UNSUPPORTED DIFFUSION MODEL {}".format(unet_path))
        raise RuntimeError(
            "ERROR: Could not detect model type of: {}\n{}".format(
                unet_path, model_detection_error_hint(unet_path, sd)
            )
        )
    return model


class SDNQLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "int8_matmul": ("BOOLEAN", {"default":True})
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"

    CATEGORY = "SDNQ"

    def load_unet(self, unet_name, int8_matmul:bool=True):
        unload_all_models()
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = load_diffusion_model(unet_path, int8_matmul)
        return (model,)

