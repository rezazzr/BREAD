from . import *
from .ct_model import CTranslateModel
from .hf_text2text_model import HFText2TextModel
from .hf_textgeneration_model import HFTextGenerationModel
from .openai_model import OpenAIModel

LANGUAGE_MODELS = {
    "openai": OpenAIModel,
    "hf_text2text": HFText2TextModel,
    "hf_textgeneration": HFTextGenerationModel,
    "ct_model": CTranslateModel,
}


def get_language_model(language_model_name):
    assert (
        language_model_name in LANGUAGE_MODELS.keys()
    ), f"Language model type {language_model_name} is not supported."
    return LANGUAGE_MODELS[language_model_name]
