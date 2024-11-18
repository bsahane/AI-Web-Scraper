import streamlit as st
from typing import Dict, List, Optional
import requests
import json
import os
import time
import socket

# Ollama configuration
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'host.docker.internal:11434')
OLLAMA_BASE_URL = f'http://{OLLAMA_HOST}'

def test_ollama_connection(base_url: Optional[str] = None) -> bool:
    """Test connection to Ollama and print debug information"""
    url = base_url or OLLAMA_BASE_URL
    urls_to_try = [
        url,
        'http://host.docker.internal:11434',  # Host machine's Ollama
        'http://localhost:11434',             # Direct localhost
    ]
    
    for test_url in urls_to_try:
        try:
            print(f"\nTesting Ollama connection at: {test_url}")
            response = requests.get(f"{test_url}/api/tags", timeout=5)
            print(f"Connection Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                print(f"Available Models: {models}")
                return True
        except Exception as e:
            print(f"Connection Error for {test_url}: {str(e)}")
    
    return False

def get_ollama_models() -> List[str]:
    """Get list of available Ollama models"""
    urls_to_try = [
        OLLAMA_BASE_URL,
        'http://host.docker.internal:11434',  # Host machine's Ollama
        'http://localhost:11434',             # Direct localhost
    ]
    
    for url in urls_to_try:
        try:
            print(f"\nTrying to get models from: {url}")
            response = requests.get(f"{url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'models' in data and data['models']:
                    models = [model['name'].split(':')[0] for model in data['models']]
                    models = list(dict.fromkeys(models))  # Remove duplicates
                    if models:
                        print(f"Found models at {url}: {models}")
                        return models
        except Exception as e:
            print(f"Error getting models from {url}: {str(e)}")
    
    print("No models found, using default model list")
    return ["llama2"]

def get_model_details() -> List[Dict]:
    """Get detailed information about available models"""
    urls_to_try = [
        OLLAMA_BASE_URL,
        'http://host.docker.internal:11434',  # Host machine's Ollama
        'http://localhost:11434',             # Direct localhost
    ]
    
    for url in urls_to_try:
        try:
            response = requests.get(f"{url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'models' in data and data['models']:
                    return data['models']
        except Exception as e:
            print(f"Error getting model details from {url}: {str(e)}")
    
    return []

def initialize_session_state():
    """Initialize session state variables for LLM configuration"""
    if 'llm_provider' not in st.session_state:
        st.session_state.llm_provider = 'ollama'
    if 'ollama_host' not in st.session_state:
        st.session_state.ollama_host = OLLAMA_BASE_URL
    if 'model_name' not in st.session_state:
        st.session_state.model_name = 'llama2'
    if 'available_models' not in st.session_state:
        st.session_state.available_models = get_ollama_models()

def get_available_models(provider: str) -> List[str]:
    """Get available models for the selected provider"""
    if provider == 'ollama':
        # Update available models
        models = get_ollama_models()
        st.session_state.available_models = models
        return models
    return []

def render_llm_config():
    """Render the LLM configuration UI"""
    with st.sidebar:
        st.subheader("🤖 LLM Configuration")
        
        # LLM Provider selection
        provider = st.selectbox(
            "Select Provider",
            options=['ollama'],
            index=0,
            key='llm_provider'
        )
        
        # Ollama configuration
        if provider == 'ollama':
            # Ollama host configuration
            ollama_host = st.text_input(
                "Ollama Host URL",
                value=st.session_state.get('ollama_host', OLLAMA_BASE_URL),
                key='ollama_host'
            )
            
            # Test connection button
            if st.button("Test Connection"):
                with st.spinner("Testing connection..."):
                    if test_ollama_connection(ollama_host):
                        st.success("Successfully connected to Ollama!")
                        # Update available models after successful connection
                        st.session_state.available_models = get_ollama_models()
                    else:
                        st.error("Failed to connect to Ollama. Please check the host URL and ensure Ollama is running.")
            
            # Model selection
            available_models = st.session_state.get('available_models', get_ollama_models())
            if available_models:
                model = st.selectbox(
                    "Select Model",
                    options=available_models,
                    index=available_models.index(st.session_state.model_name) if st.session_state.model_name in available_models else 0,
                    key='model_name'
                )
            else:
                st.warning("No models available. Please check your Ollama installation.")

def get_current_config() -> Dict:
    """Get the current LLM configuration"""
    return {
        'provider': st.session_state.llm_provider,
        'host': st.session_state.ollama_host,
        'model': st.session_state.model_name
    }

# Default model configuration
DEFAULT_MODEL = 'llama2'

# Provider configurations
PROVIDER_CONFIGS = {
    'ollama': {
        'name': 'Ollama',
        'description': 'Local LLM using Ollama',
        'models': ['llama2'],
        'default_model': 'llama2'
    }
}

# Debug information
print(f"\n=== Ollama Configuration Debug ===")
print(f"Ollama Host: {OLLAMA_HOST}")
print(f"Ollama Base URL: {OLLAMA_BASE_URL}")
print(f"Container Hostname: {socket.gethostname()}")
print(f"Container IP: {socket.gethostbyname(socket.gethostname())}")

initialize_session_state()
render_llm_config()
