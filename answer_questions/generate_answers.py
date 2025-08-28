"""
Calls LLMs and generates a coherent answer 
based on the received documents
"""
import os
import json
from typing import List
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from utils.load_config import load_config

load_dotenv(override=True)
config = load_config()

def load_prompt(prompt_name: str) -> PromptTemplate:
    """
    Function that loads the file with the prompt

    Args:
        prompt_name (str): name of the prompt

    Returns:
        (PromptTemplate): PromptTemplate object 
            from LangChain
    """
    prompt_path = os.path.join(
        os.path.dirname(__file__),
        'prompt.json'
    )
    try:
        with open(os.path.abspath(prompt_path), "r", encoding="utf-8") as f:
            prompt_dict = json.load(f)
        prompt_lc = PromptTemplate.from_template(prompt_dict[prompt_name])
        return prompt_lc

    except FileNotFoundError as e:
        raise FileNotFoundError("Prompt file not found.") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in prompt file: {e}") from e

def answer_with_docs(docs: List[Document], query: str) -> str:
    """
    Returns a structured answer based on the articles
    retrieved to better answer the made question

    Args:
        docs (List[Document]): list of langchain 
            documents retrieved
        query (str): question made by the user

    Returns:
        answer (str): answer to the received query
    """
    prompt = load_prompt(prompt_name="qa_styling")
    docs_content = "\n\n".join(doc.page_content for doc in docs)
    if config["local_or_api_llm"] == "api":
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, max_tokens=512)

    elif config["local_or_api_llm"] == "local":
        model_id = config["mistral_llm"]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        pipe = pipeline("text2text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=512)
        llm = HuggingFacePipeline(pipeline=pipe)
    else:
        raise ValueError("LLM can only be called through an"
                        "API ('api') or ran locally ('local')." 
                        f'Selected {config["local_or_api_llm"]}')

    chain = prompt | llm
    answer = chain.invoke({"question": query, "context": docs_content})
    return answer
