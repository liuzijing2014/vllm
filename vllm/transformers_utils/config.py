# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import json
import os
import time
from functools import cache, partial
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union

import huggingface_hub
from huggingface_hub import get_safetensors_metadata, hf_hub_download
from huggingface_hub import list_repo_files as hf_list_repo_files
from huggingface_hub import try_to_load_from_cache
from huggingface_hub.utils import (EntryNotFoundError, HfHubHTTPError,
                                   HFValidationError, LocalEntryNotFoundError,
                                   RepositoryNotFoundError,
                                   RevisionNotFoundError)
from transformers import GenerationConfig, PretrainedConfig
from transformers.models.auto.image_processing_auto import (
    get_image_processor_config)
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
from transformers.models.auto.tokenization_auto import get_tokenizer_config
from transformers.utils import CONFIG_NAME as HF_CONFIG_NAME

from vllm import envs
from vllm.logger import init_logger
# yapf conflicts with isort for this block
# yapf: disable
from vllm.transformers_utils.configs import (ChatGLMConfig, DeepseekVLV2Config,
                                             EAGLEConfig, JAISConfig,
                                             KimiVLConfig, MedusaConfig,
                                             MllamaConfig, MLPSpeculatorConfig,
                                             Nemotron_Nano_VL_Config,
                                             NemotronConfig, NVLM_D_Config,
                                             RWConfig, UltravoxConfig)
# yapf: enable
from vllm.transformers_utils.configs.mistral import adapt_config_dict
from vllm.transformers_utils.utils import check_gguf_file

if envs.VLLM_USE_MODELSCOPE:
    from modelscope import AutoConfig
else:
    from transformers import AutoConfig

MISTRAL_CONFIG_NAME = "params.json"

logger = init_logger(__name__)


def _get_hf_token() -> Optional[str]:
    """
    Get the HuggingFace token from environment variable.

    Returns None if the token is not set, is an empty string, 
    or contains only whitespace.
    This follows the same pattern as huggingface_hub library which
    treats empty string tokens as None to avoid authentication errors.
    """
    token = os.getenv('HF_TOKEN')
    if token and token.strip():
        return token
    return None


_CONFIG_REGISTRY_OVERRIDE_HF: dict[str, type[PretrainedConfig]] = {
    "mllama": MllamaConfig
}

_CONFIG_REGISTRY: dict[str, type[PretrainedConfig]] = {
    "chatglm": ChatGLMConfig,
    "deepseek_vl_v2": DeepseekVLV2Config,
    "kimi_vl": KimiVLConfig,
    "Llama_Nemotron_Nano_VL": Nemotron_Nano_VL_Config,
    "RefinedWeb": RWConfig,  # For tiiuae/falcon-40b(-instruct)
    "RefinedWebModel": RWConfig,  # For tiiuae/falcon-7b(-instruct)
    "jais": JAISConfig,
    "mlp_speculator": MLPSpeculatorConfig,
    "medusa": MedusaConfig,
    "eagle": EAGLEConfig,
    "nemotron": NemotronConfig,
    "NVLM_D": NVLM_D_Config,
    "ultravox": UltravoxConfig,
    **_CONFIG_REGISTRY_OVERRIDE_HF
}

_CONFIG_ATTRS_MAPPING: dict[str, str] = {
    "llm_config": "text_config",
}


class ConfigFormat(str, enum.Enum):
    AUTO = "auto"
    HF = "hf"
    MISTRAL = "mistral"


_R = TypeVar("_R")


def with_retry(
    func: Callable[[], _R],
    log_msg: str,
    max_retries: int = 2,
    retry_delay: int = 2,
) -> _R:
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error("%s: %s", log_msg, e)
                raise
            logger.error("%s: %s, retrying %d of %d", log_msg, e, attempt + 1,
                         max_retries)
            time.sleep(retry_delay)
            retry_delay *= 2

    raise AssertionError("Should not be reached")


# @cache doesn't cache exceptions
@cache
def list_repo_files(
    repo_id: str,
    *,
    revision: Optional[str] = None,
    repo_type: Optional[str] = None,
    token: Union[str, bool, None] = None,
) -> list[str]:

    def lookup_files() -> list[str]:
        # directly list files if model is local
        if (local_path := Path(repo_id)).exists():
            return [
                str(file.relative_to(local_path))
                for file in local_path.rglob('*') if file.is_file()
            ]
        # if model is remote, use hf_hub api to list files
        try:
            if envs.VLLM_USE_MODELSCOPE:
                from vllm.transformers_utils.utils import (
                    modelscope_list_repo_files)
                return modelscope_list_repo_files(repo_id,
                                                  revision=revision,
                                                  token=os.getenv(
                                                      "MODELSCOPE_API_TOKEN",
                                                      None))
            return hf_list_repo_files(repo_id,
                                      revision=revision,
                                      repo_type=repo_type,
                                      token=token)
        except huggingface_hub.errors.OfflineModeIsEnabled:
            # Don't raise in offline mode,
            # all we know is that we don't have this
            # file cached.
            return []

    return with_retry(lookup_files, "Error retrieving file list")


def file_exists(
    repo_id: str,
    file_name: str,
    *,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    token: Union[str, bool, None] = None,
) -> bool:
    file_list = list_repo_files(repo_id,
                                repo_type=repo_type,
                                revision=revision,
                                token=token)
    return file_name in file_list


# In offline mode the result can be a false negative
def file_or_path_exists(model: Union[str, Path], config_name: str,
                        revision: Optional[str]) -> bool:
    if (local_path := Path(model)).exists():
        return (local_path / config_name).is_file()

    # Offline mode support: Check if config file is cached already
    cached_filepath = try_to_load_from_cache(repo_id=model,
                                             filename=config_name,
                                             revision=revision)
    if isinstance(cached_filepath, str):
        # The config file exists in cache- we can continue trying to load
        return True

    # NB: file_exists will only check for the existence of the config file on
    # hf_hub. This will fail in offline mode.

    # Call HF to check if the file exists
    return file_exists(str(model),
                       config_name,
                       revision=revision,
                       token=_get_hf_token())


def patch_rope_scaling(config: PretrainedConfig) -> None:
    """Provide backwards compatibility for RoPE."""
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        patch_rope_scaling(text_config)

    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is not None:
        patch_rope_scaling_dict(rope_scaling)


def patch_rope_scaling_dict(rope_scaling: dict[str, Any]) -> None:
    if "rope_type" in rope_scaling and "type" in rope_scaling:
        rope_type = rope_scaling["rope_type"]
        rope_type_legacy = rope_scaling["type"]
        if rope_type != rope_type_legacy:
            raise ValueError(
                f"Found conflicts between 'rope_type={rope_type}' (modern "
                f"field) and 'type={rope_type_legacy}' (legacy field). "
                "You should only specify one of them.")

    if "rope_type" not in rope_scaling and "type" in rope_scaling:
        rope_scaling["rope_type"] = rope_scaling["type"]
        logger.info("Replacing legacy 'type' key with 'rope_type'")

    if "rope_type" not in rope_scaling:
        raise ValueError("rope_scaling should have a 'rope_type' key")

    if rope_scaling["rope_type"] == "su":
        rope_scaling["rope_type"] = "longrope"
        logger.warning("Replacing legacy rope_type 'su' with 'longrope'")
    elif rope_scaling["rope_type"] == "mrope":
        assert "mrope_section" in rope_scaling
        rope_scaling["rope_type"] = "default"
        logger.warning("Replacing legacy rope_type 'mrope' with 'default'")


def _uses_mrope(config: PretrainedConfig) -> bool:
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None:
        return False

    return "mrope_section" in rope_scaling


def uses_mrope(config: PretrainedConfig) -> bool:
    """Detect if the model with this config uses M-ROPE."""
    return _uses_mrope(config) or thinker_uses_mrope(config)


def thinker_uses_mrope(config: PretrainedConfig) -> bool:
    """Detect if the model contains a thinker config and it uses M-ROPE."""
    thinker_config = getattr(config, "thinker_config", None)
    if thinker_config is None:
        return False

    thinker_text_config = getattr(thinker_config, "text_config", None)
    if thinker_text_config is None:
        return False

    return uses_mrope(thinker_text_config)


def is_encoder_decoder(config: PretrainedConfig) -> bool:
    """Detect if the model with this config is used as an encoder/decoder."""
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return is_encoder_decoder(text_config)

    return getattr(config, "is_encoder_decoder", False)


def _maybe_remap_hf_config_attrs(config: PretrainedConfig) -> PretrainedConfig:
    """Remap config attributes to match the expected names."""
    for old_attr, new_attr in _CONFIG_ATTRS_MAPPING.items():
        if hasattr(config, old_attr):
            if not hasattr(config, new_attr):
                config.update({new_attr: getattr(config, old_attr)})
            delattr(config, old_attr)
            logger.debug("Remapped config attribute '%s' to '%s'", old_attr,
                         new_attr)
    return config


def get_config(
    model: Union[str, Path],
    trust_remote_code: bool,
    revision: Optional[str] = None,
    code_revision: Optional[str] = None,
    config_format: ConfigFormat = ConfigFormat.AUTO,
    hf_overrides_kw: Optional[dict[str, Any]] = None,
    hf_overrides_fn: Optional[Callable[[PretrainedConfig],
                                       PretrainedConfig]] = None,
    **kwargs,
) -> PretrainedConfig:
    # Separate model folder from file path for GGUF models

    is_gguf = check_gguf_file(model)
    if is_gguf:
        kwargs["gguf_file"] = Path(model).name
        model = Path(model).parent

    if config_format == ConfigFormat.AUTO:
        try:
            if is_gguf or file_or_path_exists(
                    model, HF_CONFIG_NAME, revision=revision):
                config_format = ConfigFormat.HF
            elif file_or_path_exists(model,
                                     MISTRAL_CONFIG_NAME,
                                     revision=revision):
                config_format = ConfigFormat.MISTRAL
            else:
                raise ValueError(
                    "Could not detect config format for no config file found. "
                    "Ensure your model has either config.json (HF format) "
                    "or params.json (Mistral format).")

        except Exception as e:
            error_message = (
                "Invalid repository ID or local directory specified:"
                " '{model}'.\nPlease verify the following requirements:\n"
                "1. Provide a valid Hugging Face repository ID.\n"
                "2. Specify a local directory that contains a recognized "
                "configuration file.\n"
                "   - For Hugging Face models: ensure the presence of a "
                "'config.json'.\n"
                "   - For Mistral models: ensure the presence of a "
                "'params.json'.\n"
                "3. For GGUF: pass the local path of the GGUF checkpoint.\n"
                "   Loading GGUF from a remote repo directly is not yet "
                "supported.\n").format(model=model)

            raise ValueError(error_message) from e

    if config_format == ConfigFormat.HF:
        config_dict, _ = PretrainedConfig.get_config_dict(
            model,
            revision=revision,
            code_revision=code_revision,
            token=_get_hf_token(),
            **kwargs,
        )

        # Use custom model class if it's in our registry
        model_type = config_dict.get("model_type")
        if model_type in _CONFIG_REGISTRY:
            config_class = _CONFIG_REGISTRY[model_type]
            config = config_class.from_pretrained(
                model,
                revision=revision,
                code_revision=code_revision,
                token=_get_hf_token(),
                **kwargs,
            )
        else:
            try:
                config = AutoConfig.from_pretrained(
                    model,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    code_revision=code_revision,
                    token=_get_hf_token(),
                    # some old custom model's config needs
                    # `has_no_defaults_at_init=True` to work.
                    has_no_defaults_at_init=trust_remote_code,
                    **kwargs,
                )
            except ValueError as e:
                if (not trust_remote_code
                        and "requires you to execute the configuration file"
                        in str(e)):
                    err_msg = (
                        "Failed to load the model config. If the model "
                        "is a custom model not yet available in the "
                        "HuggingFace transformers library, consider setting "
                        "`trust_remote_code=True` in LLM or using the "
                        "`--trust-remote-code` flag in the CLI.")
                    raise RuntimeError(err_msg) from e
                else:
                    raise e
        config = _maybe_remap_hf_config_attrs(config)

    elif config_format == ConfigFormat.MISTRAL:
        # This function loads a params.json config which
        # should be used when loading models in mistral format
        config_dict = _download_mistral_config_file(model, revision)
        if (max_position_embeddings :=
                config_dict.get("max_position_embeddings")) is None:
            max_position_embeddings = _maybe_retrieve_max_pos_from_hf(
                model, revision, **kwargs)
            config_dict["max_position_embeddings"] = max_position_embeddings

        config = adapt_config_dict(config_dict)
    else:
        supported_formats = [
            fmt.value for fmt in ConfigFormat if fmt != ConfigFormat.AUTO
        ]
        raise ValueError(
            f"Unsupported config format: {config_format}. "
            f"Supported formats are: {', '.join(supported_formats)}. "
            f"Ensure your model uses one of these configuration formats "
            f"or specify the correct format explicitly.")

    # Special architecture mapping check for GGUF models
    if is_gguf:
        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(
                f"Can't get gguf config for {config.model_type}.")
        model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
        config.update({"architectures": [model_type]})

    if hf_overrides_kw:
        logger.debug("Overriding HF config with %s", hf_overrides_kw)
        config.update(hf_overrides_kw)
    if hf_overrides_fn:
        logger.debug("Overriding HF config with %s", hf_overrides_fn)
        config = hf_overrides_fn(config)

    patch_rope_scaling(config)

    if trust_remote_code:
        maybe_register_config_serialize_by_value()

    return config


def try_get_local_file(model: Union[str, Path],
                       file_name: str,
                       revision: Optional[str] = 'main') -> Optional[Path]:
    file_path = Path(model) / file_name
    if file_path.is_file():
        return file_path
    else:
        try:
            cached_filepath = try_to_load_from_cache(repo_id=model,
                                                     filename=file_name,
                                                     revision=revision)
            if isinstance(cached_filepath, str):
                return Path(cached_filepath)
        except HFValidationError:
            ...
    return None


def get_hf_file_to_dict(file_name: str,
                        model: Union[str, Path],
                        revision: Optional[str] = 'main'):
    """
    Downloads a file from the Hugging Face Hub and returns
    its contents as a dictionary.

    Parameters:
    - file_name (str): The name of the file to download.
    - model (str): The name of the model on the Hugging Face Hub.
    - revision (str): The specific version of the model.

    Returns:
    - config_dict (dict): A dictionary containing
    the contents of the downloaded file.
    """

    file_path = try_get_local_file(model=model,
                                   file_name=file_name,
                                   revision=revision)

    if file_path is None:
        try:
            hf_hub_file = hf_hub_download(model, file_name, revision=revision)
        except huggingface_hub.errors.OfflineModeIsEnabled:
            return None
        except (RepositoryNotFoundError, RevisionNotFoundError,
                EntryNotFoundError, LocalEntryNotFoundError) as e:
            logger.debug("File or repository not found in hf_hub_download", e)
            return None
        except HfHubHTTPError as e:
            logger.warning(
                "Cannot connect to Hugging Face Hub. Skipping file "
                "download for '%s':",
                file_name,
                exc_info=e)
            return None
        file_path = Path(hf_hub_file)

    if file_path is not None and file_path.is_file():
        with open(file_path) as file:
            return json.load(file)

    return None


@cache
def get_pooling_config(model: str, revision: Optional[str] = 'main'):
    """
    This function gets the pooling and normalize
    config from the model - only applies to
    sentence-transformers models.

    Args:
        model (str): The name of the Hugging Face model.
        revision (str, optional): The specific version
        of the model to use. Defaults to 'main'.

    Returns:
        dict: A dictionary containing the pooling
        type and whether normalization is used.
    """

    modules_file_name = "modules.json"

    modules_dict = None
    if file_or_path_exists(model=model,
                           config_name=modules_file_name,
                           revision=revision):
        modules_dict = get_hf_file_to_dict(modules_file_name, model, revision)

    if modules_dict is None:
        return None

    logger.info("Found sentence-transformers modules configuration.")

    pooling = next((item for item in modules_dict
                    if item["type"] == "sentence_transformers.models.Pooling"),
                   None)
    normalize = bool(
        next((item for item in modules_dict
              if item["type"] == "sentence_transformers.models.Normalize"),
             False))

    if pooling:

        pooling_file_name = "{}/config.json".format(pooling["path"])
        pooling_dict = get_hf_file_to_dict(pooling_file_name, model, revision)
        pooling_type_name = next(
            (item for item, val in pooling_dict.items() if val is True), None)

        if pooling_type_name is not None:
            pooling_type_name = get_pooling_config_name(pooling_type_name)

        logger.info("Found pooling configuration.")
        return {"pooling_type": pooling_type_name, "normalize": normalize}

    return None


def get_pooling_config_name(pooling_name: str) -> Union[str, None]:
    if "pooling_mode_" in pooling_name:
        pooling_name = pooling_name.replace("pooling_mode_", "")

    if "_" in pooling_name:
        pooling_name = pooling_name.split("_")[0]

    if "lasttoken" in pooling_name:
        pooling_name = "last"

    supported_pooling_types = ['LAST', 'ALL', 'CLS', 'STEP', 'MEAN']
    pooling_type_name = pooling_name.upper()

    if pooling_type_name in supported_pooling_types:
        return pooling_type_name

    raise NotImplementedError(
        f"Pooling type {pooling_type_name} not supported")


@cache
def get_sentence_transformer_tokenizer_config(model: Union[str, Path],
                                              revision: Optional[str] = 'main'
                                              ):
    """
    Returns the tokenization configuration dictionary for a
    given Sentence Transformer BERT model.

    Parameters:
    - model (str|Path): The name of the Sentence Transformer
    BERT model.
    - revision (str, optional): The revision of the m
    odel to use. Defaults to 'main'.

    Returns:
    - dict: A dictionary containing the configuration parameters
    for the Sentence Transformer BERT model.
    """
    sentence_transformer_config_files = [
        "sentence_bert_config.json",
        "sentence_roberta_config.json",
        "sentence_distilbert_config.json",
        "sentence_camembert_config.json",
        "sentence_albert_config.json",
        "sentence_xlm-roberta_config.json",
        "sentence_xlnet_config.json",
    ]
    encoder_dict = None

    for config_file in sentence_transformer_config_files:
        if try_get_local_file(model=model,
                              file_name=config_file,
                              revision=revision) is not None:
            encoder_dict = get_hf_file_to_dict(config_file, model, revision)
            if encoder_dict:
                break

    if not encoder_dict and not Path(model).is_absolute():
        try:
            # If model is on HuggingfaceHub, get the repo files
            repo_files = list_repo_files(model,
                                         revision=revision,
                                         token=_get_hf_token())
        except Exception:
            repo_files = []

        for config_name in sentence_transformer_config_files:
            if config_name in repo_files:
                encoder_dict = get_hf_file_to_dict(config_name, model,
                                                   revision)
                if encoder_dict:
                    break

    if not encoder_dict:
        return None

    logger.info("Found sentence-transformers tokenize configuration.")

    if all(k in encoder_dict for k in ("max_seq_length", "do_lower_case")):
        return encoder_dict
    return None


def maybe_register_config_serialize_by_value() -> None:
    """Try to register HF model configuration class to serialize by value

        If trust_remote_code is set, and the model's config file specifies an
        `AutoConfig` class, then the config class is typically an instance of
        a custom class imported from the HF modules cache.

        Examples:

        >>> from transformers import AutoConfig
        >>> klass = AutoConfig.from_pretrained('meta-llama/Meta-Llama-3-8B', trust_remote_code=True)
        >>> klass.__class__ # transformers.models.llama.configuration_llama.LlamaConfig
        >>> import transformers_modules # error, not initialized
        >>> klass = AutoConfig.from_pretrained('deepseek-ai/DeepSeek-V2.5', trust_remote_code=True)
        >>> import transformers_modules # success, initialized
        >>> klass.__class__ # transformers_modules.deepseek-ai.DeepSeek-V2.5.98b11844770b2c3ffc18b175c758a803640f4e77.configuration_deepseek.DeepseekV2Config

        In the DeepSeek example, the config class is an instance of a custom
        class that is not serializable by default. This class will not be
        importable in spawned workers, and won't exist at all on
        other nodes, which breaks serialization of the config.

        In this function we tell the cloudpickle serialization library to pass
        instances of these generated classes by value instead of by reference,
        i.e. the class definition is serialized along with its data so that the
        class module does not need to be importable on the receiving end.

        See: https://github.com/cloudpipe/cloudpickle?tab=readme-ov-file#overriding-pickles-serialization-mechanism-for-importable-constructs
    """ # noqa
    try:
        import transformers_modules
        transformers_modules_available = True
    except ImportError:
        transformers_modules_available = False

    try:
        import multiprocessing
        import pickle

        import cloudpickle

        from vllm.config import VllmConfig

        # Register multiprocessing reducers to handle cross-process
        # serialization of VllmConfig objects that may contain custom configs
        # from transformers_modules
        def _reduce_config(config: VllmConfig):
            return (pickle.loads, (cloudpickle.dumps(config), ))

        multiprocessing.reducer.register(VllmConfig, _reduce_config)

        # Register transformers_modules with cloudpickle if available
        if transformers_modules_available:
            cloudpickle.register_pickle_by_value(transformers_modules)

            # ray vendors its own version of cloudpickle
            from vllm.executor.ray_utils import ray
            if ray:
                ray.cloudpickle.register_pickle_by_value(transformers_modules)

    except Exception as e:
        logger.warning(
            "Unable to register remote classes used by"
            " trust_remote_code with by-value serialization. This may"
            " lead to a later error. If remote code is not needed"
            " remove `--trust-remote-code`",
            exc_info=e)


def get_hf_image_processor_config(
    model: Union[str, Path],
    hf_token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    **kwargs,
) -> dict[str, Any]:
    # ModelScope does not provide an interface for image_processor
    if envs.VLLM_USE_MODELSCOPE:
        return dict()
    # Separate model folder from file path for GGUF models
    if check_gguf_file(model):
        model = Path(model).parent
    return get_image_processor_config(model,
                                      token=hf_token,
                                      revision=revision,
                                      **kwargs)


def get_hf_text_config(config: PretrainedConfig):
    """Get the "sub" config relevant to llm for multi modal models.
    No op for pure text models.
    """
    text_config = config.get_text_config()

    if text_config is not config:
        # The code operates under the assumption that text_config should have
        # `num_attention_heads` (among others). Assert here to fail early
        # if transformers config doesn't align with this assumption.
        assert hasattr(text_config, "num_attention_heads")

    return text_config


def try_get_generation_config(
    model: str,
    trust_remote_code: bool,
    revision: Optional[str] = None,
) -> Optional[GenerationConfig]:
    try:
        return GenerationConfig.from_pretrained(
            model,
            revision=revision,
        )
    except OSError:  # Not found
        try:
            config = get_config(
                model,
                trust_remote_code=trust_remote_code,
                revision=revision,
            )
            return GenerationConfig.from_model_config(config)
        except OSError:  # Not found
            return None


def try_get_safetensors_metadata(
    model: str,
    *,
    revision: Optional[str] = None,
):
    get_safetensors_metadata_partial = partial(
        get_safetensors_metadata,
        model,
        revision=revision,
        token=_get_hf_token(),
    )

    try:
        return with_retry(get_safetensors_metadata_partial,
                          "Error retrieving safetensors")
    except Exception:
        return None


def try_get_tokenizer_config(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    trust_remote_code: bool,
    revision: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    try:
        return get_tokenizer_config(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
    except Exception:
        return None


def _download_mistral_config_file(model, revision) -> dict:
    config_file_name = "params.json"
    config_dict = get_hf_file_to_dict(config_file_name, model, revision)
    if config_dict is None:
        raise ValueError(
            f"Failed to load mistral '{config_file_name}' config for model "
            f"{model}. Please check if the model is a mistral-format model "
            f"and if the config file exists.")
    assert isinstance(config_dict, dict)
    return config_dict


def _maybe_retrieve_max_pos_from_hf(model, revision, **kwargs) -> int:
    max_position_embeddings = 128_000
    try:
        trust_remote_code_val = kwargs.get("trust_remote_code", False)
        hf_config = get_config(model=model,
                               trust_remote_code=trust_remote_code_val,
                               revision=revision,
                               config_format=ConfigFormat.HF)
        if hf_value := hf_config.get_text_config().max_position_embeddings:
            max_position_embeddings = hf_value
    except Exception as e:
        logger.warning(
            "The params.json file is missing 'max_position_embeddings'"
            " and could not get a value from the HF config."
            " Defaulting to 128000",
            exc_info=e)

    return max_position_embeddings
