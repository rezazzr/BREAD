from .openai_model import OpenAIModel

# To register a new model backend:
# 1. Create a new file (e.g., my_model.py) with a class inheriting BaseLanguageModel
# 2. Implement batch_forward_func() and generate()
# 3. Add an entry to LANGUAGE_MODELS below
LANGUAGE_MODELS = {
    "openai": OpenAIModel,
}

# Load any local-only backends (e.g. _local_backends.py, which is gitignored)
try:
    from ._local_backends import LOCAL_MODELS
    LANGUAGE_MODELS.update(LOCAL_MODELS)
except ImportError:
    pass


def get_language_model(language_model_name):
    if language_model_name not in LANGUAGE_MODELS:
        raise ValueError(
            f"Language model type '{language_model_name}' is not supported. "
            f"Available: {list(LANGUAGE_MODELS.keys())}"
        )
    return LANGUAGE_MODELS[language_model_name]


__all__ = ["get_language_model", "LANGUAGE_MODELS", "OpenAIModel"]
