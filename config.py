import streamlit as st
from typing import Dict, List, Optional
import requests
import json
import os
import time
import socket
import subprocess

# Ollama configuration
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'localhost:11434')
OLLAMA_BASE_URL = f'http://{OLLAMA_HOST}'

# Debug information
print(f"\n=== Ollama Configuration Debug ===")
print(f"Ollama Host: {OLLAMA_HOST}")
print(f"Ollama Base URL: {OLLAMA_BASE_URL}")
print(f"Container Hostname: {socket.gethostname()}")
print(f"Container IP: {socket.gethostbyname(socket.gethostname())}")

def test_ollama_connection():
    """Test connection to Ollama and print debug information"""
    try:
        # Test basic connection
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        print(f"\n=== Ollama Connection Test ===")
        print(f"Connection Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            print(f"Available Models: {models}")
            return True
        return False
    except Exception as e:
        print(f"Connection Error: {str(e)}")
        return False

def get_ollama_models() -> List[str]:
    """Get list of available Ollama models using both API and CLI"""
    models = set()
    
    # Try API first
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'models' in data and data['models']:
                api_models = [model['name'].split(':')[0] for model in data['models']]
                models.update(api_models)
    except Exception as e:
        print(f"API Error: {str(e)}")

    # Try CLI as backup
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            # Parse CLI output, skipping header
            lines = result.stdout.strip().split('\n')[1:]
            for line in lines:
                if line.strip():
                    # First column is the model name
                    model_name = line.split()[0].split(':')[0]
                    models.add(model_name)
    except Exception as e:
        print(f"CLI Error: {str(e)}")

    # If we found models, return them
    if models:
        model_list = sorted(list(models))
        print(f"\nAvailable Ollama Models:")
        for model in model_list:
            print(f"  - {model}")
        return model_list

    # Fallback to default model
    print("No models found, using default")
    return ["llama2"]

def get_model_details() -> List[Dict]:
    """Get detailed information about available models"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'models' in data:
                return data['models']
    except Exception as e:
        print(f"Error getting model details: {str(e)}")
    return []

def get_ollama_models_with_details() -> List[str]:
    """Get list of available Ollama models with details"""
    models = get_model_details()
    model_names = [model['name'].split(':')[0] for model in models]
    return list(dict.fromkeys(model_names))

def initialize_session_state():
    """Initialize session state variables for LLM configuration"""
    if 'llm_provider' not in st.session_state:
        st.session_state.llm_provider = "Ollama"
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {}
    if 'api_urls' not in st.session_state:
        st.session_state.api_urls = {"Ollama": OLLAMA_BASE_URL}
    if 'custom_models' not in st.session_state:
        st.session_state.custom_models = {}
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "llama2"

    # Debug information
    print(f"Session state initialized:")
    print(f"Provider: {st.session_state.llm_provider}")
    print(f"API URLs: {st.session_state.api_urls}")
    print(f"Selected model: {st.session_state.selected_model}")
    print(f"Available models: {get_ollama_models_with_details()}")

def get_available_models(provider: str) -> List[str]:
    """Get available models for the selected provider"""
    if provider == "Ollama":
        api_url = st.session_state.api_urls.get("Ollama")
        models = get_ollama_models_with_details()
    else:
        models = PROVIDER_CONFIGS[provider]["default_models"]
    
    # Add custom models for the provider
    custom_models = st.session_state.custom_models.get(provider, [])
    models.extend(custom_models)
    
    # Add option to manually add a model
    models.append("+ Add Custom Model")
    return models

def render_llm_config():
    """Render the LLM configuration UI"""
    st.sidebar.title("LLM Configuration")
    
    # Provider selection
    provider = st.sidebar.selectbox(
        "Select LLM Provider",
        options=list(PROVIDER_CONFIGS.keys()),
        key="llm_provider"
    )
    
    # API URL configuration
    if PROVIDER_CONFIGS[provider]["requires_api_url"]:
        default_url = st.session_state.api_urls.get(
            provider,
            PROVIDER_CONFIGS[provider].get("default_url", "")
        )
        api_url = st.sidebar.text_input(
            f"{provider} API URL",
            value=default_url,
            type="default"
        )
        st.session_state.api_urls[provider] = api_url
    
    # API key configuration
    if PROVIDER_CONFIGS[provider]["requires_api_key"]:
        api_key = st.sidebar.text_input(
            f"{provider} API Key",
            value=st.session_state.api_keys.get(provider, ""),
            type="password"
        )
        st.session_state.api_keys[provider] = api_key
    
    # Model selection
    models = get_available_models(provider)
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=models
    )
    
    # Handle custom model addition
    if selected_model == "+ Add Custom Model":
        custom_model = st.sidebar.text_input("Enter Custom Model Name")
        if custom_model and st.sidebar.button("Add Model"):
            if provider not in st.session_state.custom_models:
                st.session_state.custom_models[provider] = []
            if custom_model not in st.session_state.custom_models[provider]:
                st.session_state.custom_models[provider].append(custom_model)
                st.experimental_rerun()
    else:
        st.session_state.selected_model = selected_model

def get_current_config() -> Dict:
    """Get the current LLM configuration"""
    provider = st.session_state.llm_provider
    config = {
        "provider": provider,
        "model": st.session_state.selected_model,
    }
    
    if PROVIDER_CONFIGS[provider]["requires_api_key"]:
        config["api_key"] = st.session_state.api_keys.get(provider)
    
    if PROVIDER_CONFIGS[provider]["requires_api_url"]:
        config["api_url"] = st.session_state.api_urls.get(provider)
    
    return config

# Default model configuration
DEFAULT_MODEL = 'llama2'

# Provider configurations
PROVIDER_CONFIGS = {
    "Ollama": {
        "default_url": OLLAMA_BASE_URL,
        "requires_api_key": False,
        "requires_api_url": True,
        "default_models": get_ollama_models_with_details() or ["llama2"],
    },
    "OpenAI": {
        "requires_api_key": True,
        "requires_api_url": False,
        "default_models": [
            "gpt-4-turbo-preview",
            "gpt-4",
            "gpt-3.5-turbo",
        ],
    },
    "Groq": {
        "requires_api_key": True,
        "requires_api_url": False,
        "default_models": [
            "mixtral-8x7b-32768",
            "llama2-70b-4096",
        ],
    },
    "Claude": {
        "requires_api_key": True,
        "requires_api_url": False,
        "default_models": [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-2.1",
        ],
    },
    "Custom OpenAI": {
        "requires_api_key": True,
        "requires_api_url": True,
        "default_models": [],
    }
}
