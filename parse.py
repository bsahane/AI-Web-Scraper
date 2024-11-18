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
import dns.resolver
import whois
import socket
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import ssl
import OpenSSL.crypto as crypto
from datetime import datetime

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

def format_list_response(response_text, query_type):
    """Format list-type responses with proper structure."""
    try:
        lines = response_text.split('\n')
        formatted_lines = []
        current_section = None
        
        # Define patterns for different types of information
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        phone_pattern = r'(?:\+\d{1,3}[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}'
        
        # Remove common verification phrases
        cleanup_patterns = [
            r'\[Verification:.*?\]',
            r'\[Confidence:.*?\]',
            r'Based on .*?, ',
            r'I found .*?:',
            r'Here are .*?:',
            r'Please note.*?\.',
        ]
        
        for pattern in cleanup_patterns:
            response_text = re.sub(pattern, '', response_text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Process each line
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip common filler phrases
            if any(phrase in line.lower() for phrase in ['i can', 'please note', 'based on', 'here are']):
                continue
            
            # Process based on query type
            if query_type == 'email':
                emails = re.findall(email_pattern, line)
                for email in emails:
                    if email not in formatted_lines:
                        formatted_lines.append(f"• {email}")
            
            elif query_type == 'phone':
                phones = re.findall(phone_pattern, line)
                for phone in phones:
                    if phone not in formatted_lines:
                        formatted_lines.append(f"• {phone}")
            
            elif query_type == 'list':
                # Remove bullet points and numbers at the start
                line = re.sub(r'^[\d\-\•\*\→\.\s]+', '', line)
                if line and line not in formatted_lines:
                    formatted_lines.append(f"• {line}")
        
        # Format the final response
        if query_type in ['email', 'phone']:
            header = "Contact Information:\n"
        else:
            header = ""
        
        # Remove duplicates while preserving order
        seen = set()
        unique_lines = []
        for line in formatted_lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        
        return header + "\n".join(unique_lines)
    
    except Exception as e:
        print(f"Error formatting list response: {str(e)}")
        return response_text

def verify_and_clean_response(response_text: str, query_type: str, llm_config: dict) -> str:
    """Verify and clean the response format using LLM."""
    try:
        llm = get_llm(llm_config)
        
        # Create verification prompt based on query type
        if query_type == 'schedule':
            verify_prompt = f"""
            Clean and verify this schedule information. Follow these rules strictly:
            1. Each entry should be on a new line
            2. Use format: "Time - Event - Location"
            3. Remove any duplicates
            4. Sort chronologically
            5. Remove any explanatory text or notes
            6. Ensure consistent time format (HH:MM AM/PM)
            7. Remove any incomplete or malformed entries
            
            Input:
            {response_text}
            
            Provide only the cleaned schedule entries, nothing else:
            """
        
        elif query_type == 'address':
            verify_prompt = f"""
            Clean and verify this address information. Follow these rules strictly:
            1. Each address should be on a new line
            2. Start each line with "•"
            3. Remove any duplicates
            4. Keep only complete addresses
            5. Remove any explanatory text
            6. Format consistently
            
            Input:
            {response_text}
            
            Provide only the cleaned address entries, nothing else:
            """
        
        elif query_type in ['email', 'phone']:
            verify_prompt = f"""
            Clean and verify this contact information. Follow these rules strictly:
            1. Each entry should be on a new line
            2. Start each line with "•"
            3. Remove any duplicates
            4. Keep only valid {query_type}s
            5. Remove any explanatory text
            6. Format consistently
            
            Input:
            {response_text}
            
            Provide only the cleaned {query_type} entries, nothing else:
            """
        
        else:  # general list
            verify_prompt = f"""
            Clean and verify this list. Follow these rules strictly:
            1. Each item should be on a new line
            2. Start each line with "•"
            3. Remove any duplicates
            4. Remove any explanatory text
            5. Keep only complete and relevant items
            6. Format consistently
            
            Input:
            {response_text}
            
            Provide only the cleaned list entries, nothing else:
            """
        
        # Get verified response
        verified = llm.predict(verify_prompt)
        
        # Add appropriate header
        headers = {
            'schedule': 'Schedule Information:',
            'address': 'Address Information:',
            'email': 'Email Addresses:',
            'phone': 'Contact Numbers:',
            'list': ''
        }
        
        header = headers.get(query_type, '')
        if header:
            verified = f"{header}\n\n{verified.strip()}"
        
        return verified.strip()
    
    except Exception as e:
        print(f"Error in verify_and_clean_response: {str(e)}")
        return response_text

def understand_query(query: str, llm_config: dict) -> dict:
    """Understand the user's query and determine the expected response format."""
    try:
        llm = get_llm(llm_config)
        understanding_prompt = f"""
        Analyze this user query and determine the following:
        Query: "{query}"
        
        1. What type of information is being requested?
        2. What specific details should be extracted?
        3. What's the best format to present this information?
        
        Return your analysis in JSON format:
        {{
            "query_type": "schedule|address|contact|list|general",
            "expected_details": ["detail1", "detail2"],
            "format_type": "table|bullets|paragraph",
            "special_requirements": ["requirement1", "requirement2"]
        }}
        """
        
        response = llm.predict(understanding_prompt)
        return json.loads(response)
    except Exception as e:
        print(f"Error in understand_query: {str(e)}")
        return {
            "query_type": "general",
            "expected_details": [],
            "format_type": "paragraph",
            "special_requirements": []
        }

def extract_information(content: str, query_understanding: dict, llm_config: dict) -> str:
    """Extract relevant information based on query understanding."""
    try:
        llm = get_llm(llm_config)
        
        # Create extraction prompt based on query understanding
        extraction_prompt = f"""
        Extract information from this content based on the following requirements:
        
        Query Type: {query_understanding['query_type']}
        Expected Details: {', '.join(query_understanding['expected_details'])}
        Format Type: {query_understanding['format_type']}
        Special Requirements: {', '.join(query_understanding['special_requirements'])}
        
        Content:
        {content}
        
        Rules:
        1. Only extract information that exists in the content
        2. Be precise and accurate
        3. Format according to the specified format type
        4. Follow any special requirements
        5. Remove any duplicate information
        """
        
        return llm.predict(extraction_prompt)
    except Exception as e:
        print(f"Error in extract_information: {str(e)}")
        return ""

def verify_and_format_response(extracted_info: str, query_understanding: dict, llm_config: dict) -> str:
    """Verify the extracted information and format it appropriately."""
    try:
        llm = get_llm(llm_config)
        
        verification_prompt = f"""
        Verify and format this information:
        
        Information:
        {extracted_info}
        
        Requirements:
        1. Query Type: {query_understanding['query_type']}
        2. Format: {query_understanding['format_type']}
        3. Expected Details: {', '.join(query_understanding['expected_details'])}
        
        Verification and Formatting Rules:
        1. Focus ONLY on the specifically requested information
        2. Remove ALL unnecessary context and explanations
        3. Use clear visual hierarchy with proper spacing
        4. For schedules:
           - Use format: "HH:MM AM - Event" (24-hour format not allowed)
           - Sort chronologically
           - Group similar events
           - Add duration if available
           - Use bold for timing
        5. Use markdown formatting:
           - ## for main headers
           - Bold for important information
           - Lists with proper indentation
        6. Keep responses concise and well-organized
        7. Remove any duplicate information
        8. Use table format when it improves readability
        
        Return ONLY the verified and formatted information, nothing else:
        """
        
        return llm.predict(verification_prompt)
    except Exception as e:
        print(f"Error in verify_and_format_response: {str(e)}")
        return extracted_info

def chat_about_content(content: str, user_query: str, llm_config: dict, images=None, url=None) -> dict:
    """Enhanced chat functionality with website metadata support."""
    try:
        # Get website metadata if URL is provided
        metadata_info = ""
        if url:
            metadata = get_website_metadata(url)
            metadata_info = format_metadata_response(metadata)
        
        # Get query understanding and process response as before
        query_understanding = understand_query(user_query, llm_config)
        extracted_info = extract_information(content, query_understanding, llm_config)
        final_response = verify_and_format_response(extracted_info, query_understanding, llm_config)
        
        # Combine metadata with final response if metadata exists
        if metadata_info:
            final_response = f"{metadata_info}\n{'=' * 50}\n\n{final_response}"
        
        # Handle special formatting cases
        if query_understanding['query_type'] == 'schedule':
            schedule_info = process_schedule_response(final_response)
            if schedule_info.get('has_table', False):
                return {
                    "text": schedule_info['text'],
                    "verified": True,
                    "has_table": True,
                    "table": schedule_info['table']
                }
        
        return {
            "text": final_response,
            "verified": True,
            "has_table": False,
            "has_images": False,
            "images": None
        }
        
    except Exception as e:
        error_msg = f"Error in chat_about_content: {str(e)}"
        print(error_msg)
        return {
            "text": "I apologize, but I encountered an error while processing your request. Please try rephrasing your question.",
            "verified": False,
            "has_table": False,
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
            
            # Skip common non-schedule lines
            if any(line.lower().startswith(word) for word in ['the', 'this', 'please', 'here', 'note:']):
                continue
                
            # Check if this is a section header
            if line.endswith(':'):
                current_section = line
                has_schedule = True
                continue
            
            # Try to parse schedule entries with various formats
            parts = []
            
            # Try different delimiters
            for delimiter in [' - ', ' – ', '|', ':', '→', '⇒']:
                if delimiter in line:
                    parts = [p.strip() for p in line.split(delimiter) if p.strip()]
                    if len(parts) >= 2:
                        break
            
            # If no delimiter found, try to parse time pattern
            if not parts:
                # Enhanced time pattern to catch more formats
                time_patterns = [
                    r'(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))',  # 9:00 AM or 9 AM
                    r'(\d{1,2}[:.]\d{2}(?:\s*hrs?)?)',  # 09:00 or 09.00 hrs
                    r'(\d{1,2}\s*o\'clock)',  # 9 o'clock
                    r'(\d{4}\s*hrs)',  # 0900 hrs
                ]
                
                for pattern in time_patterns:
                    time_match = re.search(pattern, line)
                    if time_match:
                        time = time_match.group(1)
                        rest = line.replace(time, '', 1).strip(' -:→⇒')
                        parts = [time, rest]
                        break
            
            if len(parts) >= 2:
                # Process the parts
                time = parts[0]
                event = parts[1]
                description = ' '.join(parts[2:]) if len(parts) > 2 else ""
                
                # Clean and standardize time format
                time = re.sub(r'\s+', ' ', time)
                
                # Convert 24-hour format to 12-hour format
                if 'hrs' in time.lower():
                    try:
                        time = time.lower().replace('hrs', '').strip()
                        if '.' in time:
                            time = time.replace('.', ':')
                        if ':' not in time:
                            time = f"{time[:2]}:{time[2:]}"
                        time_obj = datetime.strptime(time, '%H:%M')
                        time = time_obj.strftime('%I:%M %p').lstrip('0')
                    except:
                        pass
                
                # Add missing :00 for times like "9 AM"
                if ':' not in time and any(x in time.upper() for x in ['AM', 'PM']):
                    try:
                        time_parts = time.upper().split()
                        time = f"{time_parts[0]}:00 {time_parts[1]}"
                    except:
                        pass
                
                # Clean up AM/PM format
                time = time.upper().replace('AM', ' AM').replace('PM', ' PM')
                time = re.sub(r'\s+', ' ', time).strip()
                
                # Clean up event and description
                event = event.strip(' -:→⇒')
                description = description.strip(' -:→⇒')
                
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
                df = df.reset_index(drop=True)
            except:
                pass  # Skip sorting if times can't be parsed
            
            # Generate a clean text representation
            text_response = "Schedule Information:\n\n"
            for _, row in df.iterrows():
                time = row['Time'].strip()
                event = row['Event'].strip()
                desc = row['Description'].strip()
                
                if time:
                    text_response += time
                    if event:
                        text_response += f" - {event}"
                        if desc:
                            text_response += f" - {desc}"
                else:
                    text_response += event
                    if desc:
                        text_response += f" - {desc}"
                text_response += "\n"
            
            return {
                "has_table": True,
                "table": df,
                "text": text_response
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

def get_website_metadata(url: str) -> dict:
    """Collect comprehensive website metadata."""
    try:
        metadata = {
            "performance": {},
            "security": {},
            "dns": {},
            "headers": {},
            "seo": {}
        }
        
        # Basic request data with proper headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        start_time = datetime.now()
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        end_time = datetime.now()
        
        # Performance metrics
        metadata["performance"] = {
            "response_time": (end_time - start_time).total_seconds(),
            "page_size": len(response.content) / 1024,  # KB
            "status_code": response.status_code
        }
        
        # Parse domain
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        if not domain:
            domain = parsed_url.path
        
        # DNS information
        try:
            dns_resolver = dns.resolver.Resolver()
            dns_resolver.timeout = 5
            dns_resolver.lifetime = 5
            
            metadata["dns"] = {
                "ip_address": socket.gethostbyname(domain),
                "mx_records": [],
                "ns_records": []
            }
            
            # Get MX records
            try:
                mx_records = dns_resolver.resolve(domain, 'MX')
                metadata["dns"]["mx_records"] = [str(mx.exchange) for mx in mx_records]
            except Exception as e:
                print(f"Error getting MX records: {str(e)}")
            
            # Get NS records
            try:
                ns_records = dns_resolver.resolve(domain, 'NS')
                metadata["dns"]["ns_records"] = [str(ns) for ns in ns_records]
            except Exception as e:
                print(f"Error getting NS records: {str(e)}")
            
        except Exception as e:
            print(f"Error in DNS resolution: {str(e)}")
            metadata["dns"] = {
                "error": str(e),
                "ip_address": "N/A",
                "mx_records": [],
                "ns_records": []
            }
        
        # Security information
        try:
            if parsed_url.scheme == 'https':
                context = ssl.create_default_context()
                with socket.create_connection((domain, 443)) as sock:
                    with context.wrap_socket(sock, server_hostname=domain) as ssock:
                        cert = ssock.getpeercert()
                        metadata["security"] = {
                            "ssl_issuer": cert.get('issuer', [{'organizationName': 'N/A'}])[0].get('organizationName', 'N/A'),
                            "ssl_expires": datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z'),
                            "ssl_version": ssock.version()
                        }
            else:
                metadata["security"] = {
                    "ssl_issuer": "Not HTTPS",
                    "ssl_expires": "Not HTTPS",
                    "ssl_version": "Not HTTPS"
                }
        except Exception as e:
            print(f"Error getting SSL info: {str(e)}")
            metadata["security"] = {
                "error": str(e),
                "ssl_issuer": "Error",
                "ssl_expires": "Error",
                "ssl_version": "Error"
            }
        
        # Headers analysis
        metadata["headers"] = {
            "server": response.headers.get('Server', 'N/A'),
            "content_type": response.headers.get('Content-Type', 'N/A'),
            "cache_control": response.headers.get('Cache-Control', 'N/A')
        }
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # SEO elements
        title = soup.title.string if soup.title else ''
        meta_desc = soup.find('meta', {'name': 'description'})
        meta_desc_content = meta_desc['content'] if meta_desc else ''
        
        metadata["seo"] = {
            "title": title,
            "meta_description": meta_desc_content,
            "h1_count": len(soup.find_all('h1')),
            "h2_count": len(soup.find_all('h2')),
            "h3_count": len(soup.find_all('h3')),
            "images_total": len(soup.find_all('img')),
            "images_without_alt": len([img for img in soup.find_all('img') if not img.get('alt')]),
            "links_count": len(soup.find_all('a')),
            "meta_tags_count": len(soup.find_all('meta'))
        }
        
        return metadata
    except Exception as e:
        print(f"Error in get_website_metadata: {str(e)}")
        return {
            "error": str(e),
            "performance": {"response_time": 0, "page_size": 0, "status_code": 0},
            "security": {"ssl_issuer": "Error", "ssl_expires": "Error", "ssl_version": "Error"},
            "dns": {"ip_address": "Error", "mx_records": [], "ns_records": []},
            "headers": {"server": "Error", "content_type": "Error", "cache_control": "Error"},
            "seo": {
                "title": "",
                "meta_description": "",
                "h1_count": 0,
                "h2_count": 0,
                "h3_count": 0,
                "images_total": 0,
                "images_without_alt": 0,
                "links_count": 0,
                "meta_tags_count": 0
            }
        }

def format_metadata_response(metadata: dict) -> str:
    """Format website metadata into a readable response."""
    try:
        response = []
        
        response.append("## 📊 Website Technical Information\n")
        
        # Performance Section
        response.append("### ⚡ Performance")
        perf = metadata.get("performance", {})
        response.append(f"• Response Time: **{perf.get('response_time', 'N/A'):.2f}s**")
        response.append(f"• Page Size: **{perf.get('page_size', 'N/A'):.1f} KB**")
        response.append(f"• Status Code: **{perf.get('status_code', 'N/A')}**\n")
        
        # Security Section
        response.append("### 🔒 Security")
        sec = metadata.get("security", {})
        if isinstance(sec, dict) and "error" not in sec:
            response.append(f"• SSL Issuer: **{sec.get('ssl_issuer', {}).get(b'O', b'N/A').decode()}**")
            response.append(f"• SSL Expires: **{sec.get('ssl_expires', 'N/A')}**")
            response.append(f"• SSL Version: **{sec.get('ssl_version', 'N/A')}**\n")
        else:
            response.append("• SSL Information: Not Available\n")
        
        # DNS Section
        response.append("### 🌐 DNS Information")
        dns_info = metadata.get("dns", {})
        if isinstance(dns_info, dict) and "error" not in dns_info:
            response.append(f"• IP Address: **{dns_info.get('ip_address', 'N/A')}**")
            response.append("• MX Records:")
            for mx in dns_info.get('mx_records', [])[:3]:
                response.append(f"  - {mx}")
            response.append("• Name Servers:")
            for ns in dns_info.get('ns_records', [])[:3]:
                response.append(f"  - {ns}\n")
        else:
            response.append("• DNS Information: Not Available\n")
        
        # Headers Section
        response.append("### 📋 Headers")
        headers = metadata.get("headers", {})
        response.append(f"• Server: **{headers.get('server', 'N/A')}**")
        response.append(f"• Content Type: **{headers.get('content_type', 'N/A')}**")
        response.append(f"• Cache Control: **{headers.get('cache_control', 'N/A')}**\n")
        
        # SEO Section
        response.append("### 🔍 SEO Analysis")
        seo = metadata.get("seo", {})
        response.append(f"• Title Length: **{len(seo.get('title', '')) if seo.get('title') != 'N/A' else 'N/A'}**")
        response.append(f"• Meta Description: **{'Present' if seo.get('meta_description') != 'N/A' else 'Missing'}**")
        response.append(f"• H1 Tags: **{seo.get('h1_count', 'N/A')}**")
        response.append(f"• H2 Tags: **{seo.get('h2_count', 'N/A')}**")
        response.append(f"• H3 Tags: **{seo.get('h3_count', 'N/A')}**")
        response.append(f"• Images Total: **{seo.get('images_total', 'N/A')}**")
        response.append(f"• Images without Alt: **{seo.get('images_without_alt', 'N/A')}**")
        response.append(f"• Links Count: **{seo.get('links_count', 'N/A')}**")
        response.append(f"• Meta Tags Count: **{seo.get('meta_tags_count', 'N/A')}**\n")
        
        return "\n".join(response)
    except Exception as e:
        return f"Error formatting metadata: {str(e)}"
