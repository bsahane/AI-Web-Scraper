from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import os
import shutil
import tempfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_chrome_driver():
    """Create and configure Chrome WebDriver using Selenium Grid"""
    try:
        selenium_host = os.environ.get('SELENIUM_HOST', 'selenium-chrome')
        selenium_port = os.environ.get('SELENIUM_PORT', '4444')
        
        logger.info(f"Connecting to Selenium Grid at: http://{selenium_host}:{selenium_port}")
        
        # Set up Chrome options
        chrome_options = Options()
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--headless=new')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--start-maximized')
        chrome_options.add_argument('--ignore-certificate-errors')
        
        logger.info(f"Chrome options: {chrome_options.arguments}")
        
        # Create Remote WebDriver
        driver = webdriver.Remote(
            command_executor=f'http://{selenium_host}:{selenium_port}/wd/hub',
            options=chrome_options
        )
        
        # Set page load timeout
        driver.set_page_load_timeout(30)
        
        return driver
        
    except Exception as e:
        logger.error(f"Error creating Chrome driver: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        raise

def scrape_website(url: str) -> str:
    """Scrape content from a website"""
    try:
        logger.info(f"Starting to scrape: {url}")  # Debug log
        driver = create_chrome_driver()
        
        logger.info("Chrome driver created successfully")  # Debug log
        driver.get(url)
        
        logger.info("Page loaded successfully")  # Debug log
        page_source = driver.page_source
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(page_source, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        logger.info(f"Successfully scraped {len(text)} characters")  # Debug log
        
        driver.quit()
        return text
        
    except Exception as e:
        logger.error(f"Scraping error details: {str(e)}")  # Debug log
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")  # Debug log
        raise
    finally:
        try:
            if 'driver' in locals():
                driver.quit()
        except Exception as e:
            logger.error(f"Error closing driver: {str(e)}")  # Debug log

def extract_body_content(html_content):
    if not html_content:
        return "Failed to retrieve content"
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Get text content
    text = soup.get_text()
    
    # Clean up text
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    
    return text


def clean_body_content(body_content):
    if not body_content:
        return ""
    # Remove extra whitespace
    cleaned = ' '.join(body_content.split())
    return cleaned


def split_dom_content(dom_content, max_length=6000):
    words = dom_content.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 for space
        if current_length + word_length > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
