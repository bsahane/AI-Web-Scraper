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

def get_readable_dom(soup):
    """Convert DOM to a human-readable format"""
    readable_content = []
    
    # Get title
    if soup.title:
        readable_content.append(f"📄 Page Title: {soup.title.string.strip()}\n")
    
    # Get meta description
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc and meta_desc.get('content'):
        readable_content.append(f"📝 Description: {meta_desc['content']}\n")
    
    # Get main headings
    headings = []
    for tag in ['h1', 'h2', 'h3']:
        for heading in soup.find_all(tag):
            text = heading.get_text().strip()
            if text:
                level = int(tag[1])
                indent = "  " * (level - 1)
                headings.append(f"{indent}{'#' * level} {text}")
    if headings:
        readable_content.append("📚 Page Structure:\n" + "\n".join(headings) + "\n")
    
    # Get main content sections
    main_content = []
    content_tags = ['article', 'main', 'section', 'div']
    
    for tag in content_tags:
        for element in soup.find_all(tag, class_=lambda x: x and any(keyword in str(x).lower() for keyword in ['content', 'main', 'article', 'body'])):
            section_title = element.find(['h1', 'h2', 'h3', 'h4'])
            if section_title:
                section_title = section_title.get_text().strip()
            else:
                section_title = "Content Section"
            
            paragraphs = element.find_all('p')
            if paragraphs:
                content = "\n".join(p.get_text().strip() for p in paragraphs[:3] if p.get_text().strip())
                if content:
                    main_content.append(f"📌 {section_title}:\n{content}\n")
    
    if main_content:
        readable_content.append("📄 Main Content Sections:\n" + "\n".join(main_content))
    
    # Get navigation links
    nav_links = []
    nav_elements = soup.find_all(['nav', 'menu']) + soup.find_all(class_=lambda x: x and 'nav' in str(x).lower())
    for nav in nav_elements:
        links = nav.find_all('a')
        for link in links:
            text = link.get_text().strip()
            href = link.get('href', '')
            if text and href and not href.startswith('#'):
                nav_links.append(f"  • {text} ({href})")
    
    if nav_links:
        readable_content.append("🔗 Navigation Links:\n" + "\n".join(nav_links[:10]) + "\n")
    
    # Get forms
    forms = []
    for form in soup.find_all('form'):
        form_info = []
        form_info.append("  📝 Form Fields:")
        for input_field in form.find_all(['input', 'textarea', 'select']):
            field_type = input_field.get('type', input_field.name)
            field_name = input_field.get('name', input_field.get('id', field_type))
            if field_type not in ['hidden', 'submit']:
                form_info.append(f"    • {field_name} ({field_type})")
        if len(form_info) > 1:
            forms.extend(form_info)
    
    if forms:
        readable_content.append("📋 Forms:\n" + "\n".join(forms) + "\n")
    
    return "\n".join(readable_content)

def scrape_website(url: str) -> dict:
    """Scrape content from a website"""
    try:
        logger.info(f"Starting to scrape: {url}")
        driver = create_chrome_driver()
        
        logger.info("Chrome driver created successfully")
        driver.get(url)
        
        logger.info("Page loaded successfully")
        page_source = driver.page_source
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(page_source, 'html.parser')
        
        # Get human-readable DOM content
        dom_content = get_readable_dom(soup)
        
        # Get raw text content
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        logger.info(f"Successfully scraped {len(text)} characters")
        
        driver.quit()
        return {
            "dom_content": dom_content,
            "raw_content": text
        }
        
    except Exception as e:
        logger.error(f"Scraping error details: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise
    finally:
        try:
            if 'driver' in locals():
                driver.quit()
        except Exception as e:
            logger.error(f"Error closing driver: {str(e)}")

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
