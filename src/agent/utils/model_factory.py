import os

from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel

from dotenv import load_dotenv

load_dotenv()

models: dict[str, BaseChatModel] = {}

model_provider = os.getenv("MODEL_PROVIDER")
if not model_provider:
    raise ValueError("MODEL_PROVIDER is not set in the environment variables.")

if model_provider == "openai":

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("MODEL_PROVIDER='openai' but OpenAI API key is not set in the environment variables.")
    
    small_model = os.getenv("OPENAI_SMALL_MODEL")
    medium_model = os.getenv("OPENAI_MEDIUM_MODEL")
    if not small_model or not medium_model:
        raise ValueError("MODEL_PROVIDER='openai' but OpenAI model names are not set in the environment variables. Please set OPENAI_SMALL_MODEL and OPENAI_MEDIUM_MODEL.")
    
    models["small"] = ChatOpenAI(model=small_model)
    models["medium"] = ChatOpenAI(model=medium_model)

elif model_provider == "anthropic":

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise ValueError("MODEL_PROVIDER='anthropic' but Anthropic API key is not set in the environment variables.")
    
    small_model = os.getenv("ANTHROPIC_SMALL_MODEL")
    medium_model = os.getenv("ANTHROPIC_MEDIUM_MODEL")
    if not small_model or not medium_model:
        raise ValueError("MODEL_PROVIDER='anthropic' but Anthropic model names are not set in the environment variables. Please set ANTHROPIC_SMALL_MODEL and ANTHROPIC_MEDIUM_MODEL.")
    
    models["small"] = ChatAnthropic(
        model_name=small_model,
        max_tokens=1024,
        timeout=None,
        max_retries=2,
    )
    models["medium"] = ChatAnthropic(
        model_name=medium_model,
        max_tokens=1024,
        timeout=None,
        max_retries=2,
    )

elif model_provider == "ollama":

    small_model = os.getenv("OLLAMA_SMALL_MODEL")
    medium_model = os.getenv("OLLAMA_MEDIUM_MODEL")
    if not small_model or not medium_model:
        raise ValueError("MODEL_PROVIDER='ollama' but Ollama model names are not set in the environment variables. Please set OLLAMA_SMALL_MODEL and OLLAMA_MEDIUM_MODEL.")
    models["small"] = ChatOllama(model=small_model)
    models["medium"] = ChatOllama(model=medium_model)

else:
    raise ValueError(f"MODEL_PROVIDER={model_provider} is not a valid model provider or not currently supported.")

def get_llm(size: Literal['small', 'medium'], temperature: float = 0) -> BaseChatModel:
    """
    Retrieve a language model of the specified size and set its temperature.
    Args:
        size (Literal['small', 'medium']): The size of the language model to retrieve I.e. gpt-4o-mini vs gpt-4o.
                                           The models are set in the environment variables for a specific model provider. 
                                           Supported sizes are 'small' and 'medium'.
        temperature (float, optional): The temperature setting for the language model. 
                                       Defaults to 0.
        **kwargs: Additional keyword arguments.
    Returns:
        The language model of the specified size with the temperature set.
    Raises:
        ValueError: If the specified size is not supported or not set for the current model provider.
    """
    
    if size in models:
        model = models[size]
        model.temperature = temperature
        return models[size]
    else:
        raise ValueError(f"Model size '{size}' is not supported or not set for current model provider. Supported sizes are 'small' and 'medium'.")
    