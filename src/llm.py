"""LLM provider construction and invocation helpers."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.config import settings


def _build_hf_local() -> BaseChatModel:
    """Build a chat model backed by a local Transformers pipeline."""
    import torch
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    do_sample = settings.llm_temperature > 0
    try:
        tokenizer = AutoTokenizer.from_pretrained(settings.hf_model)
        model = AutoModelForCausalLM.from_pretrained(
            settings.hf_model,
            dtype=torch.bfloat16,
        )
        model.generation_config.max_length = None

        text_gen_pipeline = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            device=settings.hf_device,
            return_full_text=False,
        )
        text_gen_pipeline.generation_config.max_new_tokens = settings.hf_max_new_tokens
        text_gen_pipeline.generation_config.max_length = None
        text_gen_pipeline.generation_config.do_sample = do_sample
        if do_sample:
            text_gen_pipeline.generation_config.temperature = settings.llm_temperature

        llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Hugging Face model '{settings.hf_model}': {exc}"
        ) from exc

    return ChatHuggingFace(llm=llm)


def _build_gemini() -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not settings.google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to .env before using provider 'gemini'."
        )
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=settings.llm_temperature,
        google_api_key=settings.google_api_key,
    )


def _build_vllm() -> BaseChatModel:
    return ChatOpenAI(
        model=settings.hf_model,
        openai_api_key=settings.vllm_api_key,
        openai_api_base=settings.vllm_api_base,
        temperature=settings.llm_temperature,
    )


@lru_cache(maxsize=4)
def get_llm(provider: str | None = None) -> BaseChatModel:
    """Return a cached LLM instance (optionally overriding provider)."""
    provider = provider or settings.llm_provider
    if provider == "hf_local":
        return _build_hf_local()
    if provider == "gemini":
        return _build_gemini()
    if provider == "vllm":
        return _build_vllm()
    raise ValueError(f"Unknown llm_provider '{provider}'. Expected 'hf_local' | 'gemini' | 'vllm'.")


def invoke_llm(prompt: str, provider: str | None = None) -> str:
    """Invoke an LLM (optionally overriding provider) and return text."""
    response = get_llm(provider=provider).invoke([HumanMessage(content=prompt)])
    return response.content if isinstance(response.content, str) else str(response.content)
