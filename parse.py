from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import requests
import json
import time
from typing import List

# Set Ollama host from environment or use localhost as default
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
print(f"Initializing with Ollama host: {OLLAMA_HOST}")  # Debug log

def get_ollama_models(api_url: str) -> List[str]:
    """Fetch available models from Ollama"""
    try:
        print(f"Attempting to connect to Ollama at: {api_url}")  # Debug log
        
        # Try the list endpoint first
        list_url = f"{api_url}/api/list"
        print(f"Trying list endpoint: {list_url}")
        
        list_response = requests.get(list_url, timeout=10)
        print(f"List endpoint response status: {list_response.status_code}")
        print(f"List endpoint response: {list_response.text}")  # Debug log
        
        if list_response.status_code == 200:
            data = list_response.json()
            models = data.get('models', [])
            if models:
                model_names = [model['name'] for model in models]
                print(f"Found models: {model_names}")  # Debug log
                return model_names
        
        # If list endpoint didn't work, try tags endpoint
        tags_url = f"{api_url}/api/tags"
        print(f"Trying tags endpoint: {tags_url}")
        
        tags_response = requests.get(tags_url, timeout=10)
        print(f"Tags endpoint response status: {tags_response.status_code}")
        print(f"Tags endpoint response: {tags_response.text}")  # Debug log
        
        if tags_response.status_code == 200:
            data = tags_response.json()
            models = data.get('models', [])
            if models:
                model_names = [model['name'] for model in models]
                print(f"Found models: {model_names}")  # Debug log
                return model_names
        
        print("No models found, using default")  # Debug log
        return ["llama2"]  # Default fallback
        
    except Exception as e:
        print(f"Error details: {str(e)}")  # Debug log
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")  # Debug log
        return ["llama2"]  # Default fallback

def get_llm(config: dict):
    """Get the appropriate LLM based on configuration"""
    provider = config["provider"]
    model = config["model"]
    
    if provider == "Ollama":
        return OllamaLLM(
            base_url=config["api_url"],
            model=model
        )
    elif provider == "OpenAI":
        return ChatOpenAI(
            api_key=config["api_key"],
            model=model
        )
    elif provider == "Groq":
        return ChatGroq(
            api_key=config["api_key"],
            model=model
        )
    elif provider == "Claude":
        return ChatAnthropic(
            api_key=config["api_key"],
            model=model
        )
    elif provider == "Custom OpenAI":
        return ChatOpenAI(
            api_key=config["api_key"],
            base_url=config["api_url"],
            model=model
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def get_available_models():
    """Fetch available models from Ollama"""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries} to fetch models from: {OLLAMA_HOST}")
            
            models = get_ollama_models(OLLAMA_HOST)
            if models:
                return models
            
            # If no models found, wait before retrying
            if attempt < max_retries - 1:
                print(f"No models found, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            
            print("No models found after all attempts, using default")
            return ["llama2"]  # Default fallback
            
        except Exception as e:
            print(f"Error during attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached, using default model")
                return ["llama2"]  # Default fallback

def analyze_content(content: str, config: dict) -> str:
    """Analyze the content using the configured LLM"""
    llm = get_llm(config)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that analyzes web content and provides comprehensive summaries."),
        ("user", "Please analyze the following web content and provide a comprehensive summary focusing on the main points "
                "and key information. Format your response in a clear, structured way.\n\n"
                "Content: {content}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({"content": content})

def extract_specific_info(content: str, query: str, config: dict) -> str:
    """Extract specific information from content based on the query"""
    llm = get_llm(config)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that extracts specific information from web content based on user queries."),
        ("user", "Extract the following information from this web content: {query}\n\n"
                "Please focus only on the requested information and format your response clearly.\n\n"
                "Content: {content}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({
        "content": content,
        "query": query
    })

def parse_content(content, config):
    try:
        # Initialize LLM with the correct host and selected model
        llm = get_llm(config)
        
        # Create and run the prompt
        template = (
            "You are tasked with extracting specific information from the following text content: {content}. "
            "Please analyze the content and provide a concise summary focusing on the main points and key information. "
            "Format your response in a clear, structured way."
        )
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm
        
        # Process the content
        response = chain.invoke({"content": content})
        return str(response)
    
    except Exception as e:
        return f"Error analyzing content: {str(e)}"

# Keep the original parse_with_ollama function for compatibility
def parse_with_ollama(dom_chunks, parse_description, config):
    try:
        llm = get_llm(config)
        
        detailed_template = (
            "You are tasked with extracting specific information from the following text content: {dom_content}. "
            "Please follow these instructions carefully: \n\n"
            "1. **Extract Information:** Only extract the information that directly matches the provided description: {parse_description}. "
            "2. **No Extra Content:** Do not include any additional text, comments, or explanations in your response. "
            "3. **Empty Response:** If no information matches the description, return an empty string ('')."
            "4. **Direct Data Only:** Your output should contain only the data that is explicitly requested, with no other text."
        )
        
        prompt = ChatPromptTemplate.from_template(detailed_template)
        chain = prompt | llm

        parsed_results = []

        for i, chunk in enumerate(dom_chunks, start=1):
            response = chain.invoke(
                {"dom_content": chunk, "parse_description": parse_description}
            )
            print(f"Parsed batch: {i} of {len(dom_chunks)}")
            parsed_results.append(str(response))

        return "\n".join(parsed_results)
    
    except Exception as e:
        return f"Error parsing content: {str(e)}"
