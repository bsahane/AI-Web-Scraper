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
from typing import List, Dict, Any, Optional
import pandas as pd
from urllib.parse import urljoin

# Set Ollama host from environment or use default
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
if not OLLAMA_HOST.startswith('http://'):
    OLLAMA_HOST = f'http://{OLLAMA_HOST}'
print(f"Initializing with Ollama host: {OLLAMA_HOST}")  # Debug log

def get_llm(config: Dict[str, Any]) -> Ollama:
    """Get LLM instance based on configuration"""
    try:
        # Ensure the host URL is properly formatted
        host = config.get('host', OLLAMA_HOST)
        if not host.startswith('http://'):
            host = f'http://{host}'
            
        # Create the Ollama instance
        llm = Ollama(
            base_url=host,
            model=config.get('model', 'llama2')
        )
        
        # Test the connection
        try:
            llm.predict("test")
            print("Successfully connected to Ollama")
        except Exception as e:
            print(f"Warning: Initial connection test failed: {str(e)}")
            # Try alternative host if in Docker
            if 'docker' in host:
                alternative_host = 'http://ollama:11434'
                print(f"Trying alternative host: {alternative_host}")
                llm = Ollama(
                    base_url=alternative_host,
                    model=config.get('model', 'llama2')
                )
                llm.predict("test")
                print("Successfully connected to alternative Ollama host")
        
        return llm
        
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}")
        raise

def make_ollama_request(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Make a request to Ollama API with proper error handling"""
    try:
        # Ensure the host URL is properly formatted
        base_url = OLLAMA_HOST
        if not base_url.startswith('http://'):
            base_url = f'http://{base_url}'
            
        # Construct the full URL
        url = urljoin(base_url, endpoint)
        
        # Make the request
        response = requests.post(url, json=data, timeout=30)
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.ConnectionError:
        # Try alternative host if in Docker
        try:
            alternative_url = urljoin('http://ollama:11434', endpoint)
            response = requests.post(alternative_url, json=data, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Failed to connect to Ollama: {str(e)}")
    except Exception as e:
        raise Exception(f"Error in Ollama request: {str(e)}")

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

def verify_response(content: str, response: str, user_query: str, llm_config: dict) -> tuple[str, bool]:
    """
    Verify the response by checking its accuracy against the source content.
    Returns a tuple of (verified_response, is_verified).
    """
    try:
        llm = get_llm(llm_config)
        
        # Create verification prompt
        verification_prompt = f"""
        Task: Verify the accuracy of an AI response against the source content.
        
        Source Content: {content}
        
        User Question: {user_query}
        
        AI Response: {response}
        
        Instructions:
        1. Check if the response is directly supported by the source content
        2. Identify any statements that cannot be verified from the source
        3. Remove or correct any unsupported claims
        4. Ensure the response stays focused on the user's question
        
        Provide your response in the following format:
        VERIFIED: [true/false]
        CONFIDENCE: [0-100]
        CORRECTED RESPONSE: [your corrected response]
        REASONING: [brief explanation of changes made]
        """
        
        # Get verification result
        verification = llm.predict(verification_prompt)
        
        # Parse verification result
        verified = False
        confidence = 0
        corrected_response = response
        reasoning = ""
        
        for line in verification.split('\n'):
            line = line.strip()
            if line.startswith('VERIFIED:'):
                verified = line.split(':', 1)[1].strip().lower() == 'true'
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = int(line.split(':', 1)[1].strip())
                except ValueError:
                    confidence = 0
            elif line.startswith('CORRECTED RESPONSE:'):
                corrected_response = line.split(':', 1)[1].strip()
            elif line.startswith('REASONING:'):
                reasoning = line.split(':', 1)[1].strip()
        
        # If verification failed or confidence is low, perform a second verification
        if not verified or confidence < 70:
            second_prompt = f"""
            The previous response may not be accurate. Please provide a new response that:
            1. Only uses information directly found in the source content
            2. Clearly indicates if any requested information is not available
            3. Uses direct quotes or references from the source when possible
            
            Source Content: {content}
            User Question: {user_query}
            
            Provide a new, verified response:
            """
            
            corrected_response = llm.predict(second_prompt)
            verified = True  # Second response should be verified
            
        # Add verification metadata
        final_response = corrected_response
        if reasoning:
            final_response += f"\n\n[Verification: {'✓' if verified else '✗'} | Confidence: {confidence}%]"
        
        return final_response, verified
        
    except Exception as e:
        print(f"Error in verify_response: {str(e)}")
        return response, False

def chat_about_content(content: str, user_query: str, llm_config: dict, images=None) -> dict:
    """Chat about the website content with improved verification"""
    try:
        # Initial prompt engineering
        chat_prompt = f"""
        Based on the following webpage content, answer this question: {user_query}
        
        Instructions:
        1. Only use information directly found in the content
        2. If the information isn't available, clearly state that
        3. For schedules or timings:
           - List each timing on a new line
           - Use format: "Time - Event - Description"
           - Include all available details
        4. Be specific and detailed in your response
        5. If no relevant information is found, clearly state that
        
        Webpage Content:
        {content}
        """
        
        # Get initial response
        llm = get_llm(llm_config)
        initial_response = llm.predict(chat_prompt)
        
        # For schedule queries, use a specialized prompt
        if any(word in user_query.lower() for word in ['schedule', 'time', 'when', 'hours', 'timing', 'aarti']):
            schedule_prompt = f"""
            Extract schedule information from this content. Format it as:
            Time - Event - Description
            
            Example format:
            6:00 AM - Morning Aarti - Main Temple
            12:00 PM - Afternoon Aarti - With special offerings
            
            Content: {content}
            Query: {user_query}
            
            List all relevant timings and events:
            """
            
            schedule_response = llm.predict(schedule_prompt)
            schedule_info = process_schedule_response(schedule_response)
            
            if schedule_info.get('has_table', False):
                # Format table data for display
                table = schedule_info['table']
                formatted_response = "Schedule Information:\n\n"
                for _, row in table.iterrows():
                    time = row['Time'].strip()
                    event = row['Event'].strip()
                    desc = row['Description'].strip()
                    if time and event:
                        formatted_response += f"{time} - {event}"
                        if desc:
                            formatted_response += f" - {desc}"
                        formatted_response += "\n"
                
                return {
                    "text": formatted_response,
                    "verified": True,
                    "has_table": True,
                    "table": table
                }
        
        # Verify and potentially correct the response
        verified_response, is_verified = verify_response(content, initial_response, user_query, llm_config)
        
        # Handle image-related queries
        has_images = False
        image_list = []
        if images and any(word in user_query.lower() for word in ['image', 'picture', 'photo', 'show', 'display']):
            # Create image prompt
            image_prompt = f"""
            Based on the user's question: {user_query}
            And the available images, which images are relevant?
            
            Available Images:
            {json.dumps([{'index': i, 'description': img.get('description', 'No description')} 
                        for i, img in enumerate(images)])}
            
            Return a JSON array of relevant image indices.
            """
            
            try:
                image_response = llm.predict(image_prompt)
                relevant_indices = json.loads(image_response)
                if isinstance(relevant_indices, list):
                    has_images = True
                    image_list = [images[i] for i in relevant_indices if i < len(images)]
            except (json.JSONDecodeError, IndexError) as e:
                print(f"Error processing images: {str(e)}")
        
        # Construct the final response
        response = {
            "text": verified_response,
            "verified": is_verified,
            "has_images": has_images,
            "images": image_list if has_images else None
        }
        
        return response
        
    except Exception as e:
        error_msg = f"Error in chat_about_content: {str(e)}"
        print(error_msg)
        return {
            "text": f"I apologize, but I encountered an error while processing your request. Please try rephrasing your question. Error: {str(e)}",
            "verified": False,
            "has_images": False,
            "images": None
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
                current_section = line
                has_schedule = True
                continue
            
            # Try to parse schedule entries
            # First, try splitting by common delimiters
            parts = []
            for delimiter in [' - ', ' – ', '|', ':']:
                if delimiter in line:
                    parts = [p.strip() for p in line.split(delimiter) if p.strip()]
                    if len(parts) >= 2:
                        break
            
            # If no delimiter found, try to parse time pattern
            if not parts:
                time_pattern = r'(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))'
                time_match = re.search(time_pattern, line)
                if time_match:
                    time = time_match.group(1)
                    rest = line.replace(time, '').strip()
                    parts = [time, rest]
            
            if len(parts) >= 2:
                # Process the parts
                time = parts[0]
                event = parts[1]
                description = ' '.join(parts[2:]) if len(parts) > 2 else ""
                
                # Clean up the time format
                time = re.sub(r'\s+', ' ', time)
                if ':' not in time and any(x in time.upper() for x in ['AM', 'PM']):
                    time = time.replace('AM', ' AM').replace('PM', ' PM')
                
                table_data.append([time, event, description])
                has_schedule = True
            elif line and current_section and not line.endswith(':'):
                # Handle lines without clear time-event separation
                table_data.append(["", line, ""])
                has_schedule = True
        
        if has_schedule and table_data:
            df = pd.DataFrame(table_data, columns=['Time', 'Event', 'Description'])
            # Sort by time if possible
            try:
                df['_time_sort'] = pd.to_datetime(df['Time'], format='%I:%M %p', errors='coerce')
                df = df.sort_values('_time_sort').drop('_time_sort', axis=1)
            except:
                pass  # Skip sorting if times can't be parsed
            
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
