from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, SystemMessage
import os
import requests
import json
import time
import ollama
import re
from typing import List
import pandas as pd

# Set Ollama host from environment or use localhost as default
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
print(f"Initializing with Ollama host: {OLLAMA_HOST}")  # Debug log

def get_llm(config):
    """Get LLM instance based on configuration"""
    try:
        return Ollama(
            base_url=OLLAMA_HOST,
            model=config.get('model', 'llama2')
        )
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}")
        raise

def format_response_with_table(text):
    """Convert response to include pandas DataFrame for tables"""
    # Check if the response contains table-like data
    if any(keyword in text.lower() for keyword in ['schedule', 'timing', 'table', 'list']):
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        table_data = []
        regular_text = []
        headers = None
        
        for line in lines:
            # Skip lines that are clearly not table data
            if len(line) < 3 or line.startswith('#') or line.startswith('>'):
                regular_text.append(line)
                continue
                
            # Try to split line by common delimiters
            parts = [p.strip() for p in re.split(r'[:|,-]\s*', line) if p.strip()]
            
            if len(parts) > 1:
                if headers is None:
                    headers = parts
                else:
                    # Ensure row has same number of columns as headers
                    while len(parts) < len(headers):
                        parts.append("")
                    table_data.append(parts[:len(headers)])
            else:
                regular_text.append(line)
        
        if headers and table_data:
            df = pd.DataFrame(table_data, columns=headers)
            return {
                "has_table": True,
                "table": df,
                "text": "\n".join(regular_text)
            }
    
    return {
        "has_table": False,
        "text": text
    }

def clean_verified_response(response: str) -> str:
    """Clean up verification metadata from the response."""
    # List of verification-related phrases to remove
    cleanup_patterns = [
        r"The response provided accurately answers.*?\n",
        r"All stated facts match.*?\n",
        r"The response is accurate.*?\n",
        r"Since the response is accurate.*?\n",
        r"I will return the original response:.*?\n",
        r"This response is accurate.*?\n",
        r"The information provided.*?\n",
        r"After verifying.*?\n",
        r"Upon verification.*?\n",
        r"The response correctly.*?\n"
    ]
    
    cleaned = response
    for pattern in cleanup_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove extra newlines
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned

def verify_response(content: str, response: str, user_query: str, llm_config: dict) -> str:
    """Verify the response by asking LLM to check its accuracy against the content."""
    verification_prompt = f"""You are a fact-checker. Verify if the following response accurately answers the user's question based on the provided content.
    
    Content: {content}
    
    User Question: {user_query}
    
    Response to Verify: {response}
    
    Instructions:
    1. Check if the response contains any information not present in the content
    2. Verify if all stated facts match the content exactly
    3. Ensure the response directly answers the user's question
    4. If you find any inaccuracies, provide the correct information from the content
    5. If the response is accurate but incomplete, add missing relevant information
    6. DO NOT include any verification metadata in your response
    7. Just provide the final, corrected answer
    
    Return format:
    - If accurate: Return ONLY the verified information without any verification statements
    - If needs correction: Return ONLY the corrected information without any verification statements
    - If wrong: Return ONLY the accurate information without any verification statements"""
    
    try:
        verification = ollama.chat(
            model=llm_config.get('model', 'llama2'),
            messages=[{"role": "user", "content": verification_prompt}],
            stream=False
        )
        
        verified_response = verification['message']['content']
        
        # Clean up the response
        cleaned_response = clean_verified_response(verified_response)
        
        # If the cleaned response is too short, use the original
        if len(cleaned_response.strip()) < 10:
            return response
            
        return cleaned_response
        
    except Exception as e:
        print(f"Error in verification: {str(e)}")
        return response  # Return original response if verification fails

def chat_about_content(content: str, user_query: str, llm_config: dict, images=None) -> dict:
    """Chat about the website content"""
    try:
        # Check if query is about images
        image_keywords = ['image', 'picture', 'photo', 'show', 'display', 'icon', 'logo']
        is_image_query = any(keyword in user_query.lower() for keyword in image_keywords)
        
        # Check if query is about schedules
        schedule_keywords = ['schedule', 'timing', 'time', 'aarti', 'puja', 'darshan']
        is_schedule_query = any(keyword in user_query.lower() for keyword in schedule_keywords)
        
        system_prompt = """You are a helpful assistant analyzing a website's content. 
        Follow these guidelines:
        1. Provide clear, concise answers based ONLY on the content provided
        2. If the information isn't available in the content, say so honestly
        3. When presenting information with images:
           - Describe what each relevant image shows
           - Include image descriptions and alt text when available
           - Reference images by their descriptions
        4. When presenting schedules, timings, or lists:
           - Present each item on a new line
           - Include all available details
           - Maintain the original formatting and order
        5. Always provide a meaningful response, even if the exact information isn't found
        6. Be precise and accurate - only include information that is explicitly stated in the content"""
        
        if is_image_query and images:
            system_prompt += """
            For image-related queries:
            - Focus on describing relevant images
            - Include image descriptions and context
            - Mention if images show the requested content
            - Format each image reference clearly"""
        
        if is_schedule_query:
            system_prompt += """
            For schedule-related queries:
            - Extract ALL schedule information
            - Format in a clear, structured way
            - Include times, events, and descriptions
            - Separate different schedule types clearly"""
        
        # Create content text with images if available
        content_text = f"Website content:\n{content}\n"
        if images:
            content_text += "\nAvailable images:\n"
            for img in images:
                content_text += f"- {img.get('description', 'No description')} (Alt: {img.get('alt_text', 'No alt text')})\n"
        content_text += f"\nUser question: {user_query}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_text}
        ]
        
        # Get initial response
        response = ollama.chat(
            model=llm_config.get('model', 'llama2'),
            messages=messages,
            stream=False
        )
        
        response_text = response['message']['content']
        
        # Verify the response
        verified_response = verify_response(content, response_text, user_query, llm_config)
        
        # Process schedule information if present
        if is_schedule_query:
            schedule_response = process_schedule_response(verified_response)
            if schedule_response["has_table"]:
                return schedule_response
        
        # Process image information if present
        if is_image_query and images:
            relevant_images = []
            for img in images:
                img_text = f"{img.get('description', '')} {img.get('alt_text', '')} {img.get('title', '')}".lower()
                query_terms = user_query.lower().split()
                
                if any(term in img_text for term in query_terms):
                    relevant_images.append(img)
            
            if relevant_images:
                return {
                    "has_images": True,
                    "images": relevant_images,
                    "text": verified_response
                }
        
        # Return normal response
        return {
            "has_images": False,
            "text": verified_response
        }
        
    except Exception as e:
        import traceback
        print(f"Error in chat_about_content: {str(e)}")
        print(traceback.format_exc())
        return {
            "has_images": False,
            "text": f"I encountered an error while processing your request: {str(e)}"
        }

def process_schedule_response(response_text):
    """Process schedule information from response."""
    try:
        lines = response_text.split('\n')
        table_data = []
        current_section = None
        has_schedule = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a section header
            if any(line.lower().startswith(header) for header in ['daily', 'weekly', 'monthly', 'yearly', 'schedule:', 'timing:', 'aarti:']):
                if current_section != line:
                    current_section = line
                    has_schedule = True
                    table_data.append(["", f"**{line}**", ""])
                continue
            
            # Try to parse schedule entries
            parts = [p.strip() for p in re.split(r'\||[-–]', line) if p.strip()]
            
            if len(parts) >= 2:
                if len(parts) >= 3:
                    time, event, description = parts[0], parts[1], ' - '.join(parts[2:])
                else:
                    time, event = parts[0], parts[1]
                    description = ""
                
                time = re.sub(r'\s+', ' ', time)
                table_data.append([time, event, description])
                has_schedule = True
            elif line and current_section and not line.endswith(':'):
                table_data.append(["", line, ""])
                has_schedule = True
        
        if has_schedule and table_data:
            df = pd.DataFrame(table_data, columns=['Time', 'Event', 'Description'])
            return {
                "has_table": True,
                "table": df,
                "text": response_text
            }
        
        return {
            "has_table": False,
            "text": response_text
        }
    except Exception as e:
        print(f"Error processing schedule: {str(e)}")
        return {
            "has_table": False,
            "text": response_text
        }

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
    
    prompt = PromptTemplate.from_messages([
        {"role": "system", "content": "You are a helpful assistant that analyzes web content and provides comprehensive summaries."},
        {"role": "user", "content": f"""Please analyze the following web content and provide a comprehensive summary focusing on the main points 
                and key information. Format your response in a clear, structured way.\n\n
                Content: {content}"""}
    ])
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    return chain({"content": content})

def extract_specific_info(content: str, query: str, config: dict) -> str:
    """Extract specific information from content based on the query"""
    llm = get_llm(config)
    
    prompt = PromptTemplate.from_messages([
        {"role": "system", "content": "You are a helpful assistant that extracts specific information from web content based on user queries."},
        {"role": "user", "content": f"""Extract the following information from this web content: {query}\n\n
                Please focus only on the requested information and format your response clearly.\n\n
                Content: {content}"""}
    ])
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    return chain({"content": content, "query": query})

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
        prompt = PromptTemplate.from_template(template)
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Process the content
        response = chain({"content": content})
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
        
        prompt = PromptTemplate.from_template(detailed_template)
        chain = LLMChain(llm=llm, prompt=prompt)

        parsed_results = []

        for i, chunk in enumerate(dom_chunks, start=1):
            response = chain({"dom_content": chunk, "parse_description": parse_description})
            print(f"Parsed batch: {i} of {len(dom_chunks)}")
            parsed_results.append(str(response))

        return "\n".join(parsed_results)
    
    except Exception as e:
        return f"Error parsing content: {str(e)}"
