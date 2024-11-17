from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import os
import shutil
import tempfile
import logging
import requests
from urllib.parse import urljoin, urlparse
from pathlib import Path
import hashlib

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
    
    # Get main headings and their content
    headings = []
    for tag in ['h1', 'h2', 'h3']:
        for heading in soup.find_all(tag):
            text = heading.get_text().strip()
            if text:
                level = int(tag[1])
                indent = "  " * (level - 1)
                headings.append(f"{indent}{'#' * level} {text}")
                
                # Get content under this heading
                next_element = heading.find_next_sibling()
                while next_element and next_element.name not in ['h1', 'h2', 'h3']:
                    if next_element.name in ['table', 'ul', 'ol']:
                        if next_element.name == 'table':
                            # Extract table content
                            rows = []
                            for row in next_element.find_all('tr'):
                                cols = [col.get_text().strip() for col in row.find_all(['th', 'td'])]
                                if any(cols):  # Only add non-empty rows
                                    rows.append(" | ".join(cols))
                            if rows:
                                headings.append(indent + "  Table:")
                                headings.extend(indent + "    " + row for row in rows)
                        else:
                            # Extract list content
                            items = [item.get_text().strip() for item in next_element.find_all('li')]
                            if items:
                                headings.append(indent + "  List:")
                                headings.extend(indent + "    • " + item for item in items)
                    elif next_element.name == 'p':
                        text = next_element.get_text().strip()
                        if text:
                            headings.append(indent + "  " + text)
                    next_element = next_element.find_next_sibling()
    
    if headings:
        readable_content.append("📚 Page Structure:\n" + "\n".join(headings) + "\n")
    
    # Get tables that might be schedules
    tables = soup.find_all('table')
    schedule_keywords = ['schedule', 'timing', 'aarti', 'puja', 'darshan', 'time']
    for table in tables:
        # Check if table or its container has schedule-related text
        table_text = table.get_text().lower()
        table_container = table.find_parent(['div', 'section'])
        container_text = table_container.get_text().lower() if table_container else ''
        
        if any(keyword in table_text or keyword in container_text for keyword in schedule_keywords):
            rows = []
            for row in table.find_all('tr'):
                cols = [col.get_text().strip() for col in row.find_all(['th', 'td'])]
                if any(cols):  # Only add non-empty rows
                    rows.append(" | ".join(cols))
            if rows:
                readable_content.append("📅 Schedule/Timing Information:\n" + "\n".join(rows) + "\n")
    
    # Get main content sections with better handling of lists and tables
    main_content = []
    content_tags = ['article', 'main', 'section', 'div']
    
    for tag in content_tags:
        for element in soup.find_all(tag, class_=lambda x: x and any(keyword in str(x).lower() for keyword in ['content', 'main', 'article', 'body', 'schedule', 'timing'])):
            section_title = element.find(['h1', 'h2', 'h3', 'h4'])
            if section_title:
                section_title = section_title.get_text().strip()
            else:
                section_title = "Content Section"
            
            content_parts = []
            
            # Get paragraphs
            paragraphs = element.find_all('p')
            if paragraphs:
                content = "\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
                if content:
                    content_parts.append(content)
            
            # Get lists
            lists = element.find_all(['ul', 'ol'])
            for lst in lists:
                items = [f"• {item.get_text().strip()}" for item in lst.find_all('li')]
                if items:
                    content_parts.append("\n".join(items))
            
            # Get tables
            tables = element.find_all('table')
            for table in tables:
                rows = []
                for row in table.find_all('tr'):
                    cols = [col.get_text().strip() for col in row.find_all(['th', 'td'])]
                    if any(cols):
                        rows.append(" | ".join(cols))
                if rows:
                    content_parts.append("\n".join(rows))
            
            if content_parts:
                main_content.append(f"📌 {section_title}:\n" + "\n".join(content_parts) + "\n")
    
    if main_content:
        readable_content.append("📄 Main Content Sections:\n" + "\n".join(main_content))
    
    return "\n".join(readable_content)

def is_valid_image_url(url):
    """Check if URL points to a valid image."""
    try:
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        parsed = urlparse(url)
        return any(parsed.path.lower().endswith(ext) for ext in image_extensions)
    except:
        return False

def download_image(url, base_url, save_dir):
    """Download image and return local path."""
    try:
        # Make URL absolute if it's relative
        if not bool(urlparse(url).netloc):
            url = urljoin(base_url, url)
        
        # Create hash of URL for unique filename
        url_hash = hashlib.md5(url.encode()).hexdigest()
        
        # Get file extension from URL
        ext = os.path.splitext(urlparse(url).path)[1]
        if not ext:
            ext = '.jpg'  # Default extension
        
        # Create filename and path
        filename = f"image_{url_hash}{ext}"
        filepath = os.path.join(save_dir, filename)
        
        # Skip if already downloaded
        if os.path.exists(filepath):
            return filepath
        
        # Download image
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        
        # Check if content type is image
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            return None
        
        # Save image
        with open(filepath, 'wb') as f:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, f)
        
        return filepath
    except Exception as e:
        logger.error(f"Error downloading image {url}: {str(e)}")
        return None

def extract_images(soup, base_url, save_dir):
    """Extract images from the webpage."""
    images = []
    try:
        # Create images directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Find all image elements
        img_tags = soup.find_all('img')
        for img in img_tags:
            # Get image URL
            img_url = img.get('src') or img.get('data-src')
            if not img_url:
                continue
            
            # Check if it's a valid image URL
            if not is_valid_image_url(img_url):
                continue
            
            # Get alt text and title
            alt_text = img.get('alt', '').strip()
            title = img.get('title', '').strip()
            description = alt_text or title or "No description available"
            
            # Download image
            local_path = download_image(img_url, base_url, save_dir)
            if local_path:
                images.append({
                    'url': img_url,
                    'local_path': local_path,
                    'description': description,
                    'alt_text': alt_text,
                    'title': title
                })
        
        logger.info(f"Extracted {len(images)} images from the webpage")
    except Exception as e:
        logger.error(f"Error extracting images: {str(e)}")
    
    return images

def scrape_website(url: str) -> dict:
    """Scrape content from a website"""
    try:
        driver = create_chrome_driver()
        logger.info("Chrome driver created successfully")
        
        # Create temporary directory for images
        temp_dir = tempfile.mkdtemp()
        images_dir = os.path.join(temp_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        try:
            driver.get(url)
            logger.info("Page loaded successfully")
            
            # Wait for dynamic content
            driver.implicitly_wait(5)
            
            # Get page source and create soup
            html_content = driver.page_source
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract images
            images = extract_images(soup, url, images_dir)
            
            # Get readable DOM content
            dom_content = get_readable_dom(soup)
            
            # Extract and clean body content
            body_content = extract_body_content(html_content)
            cleaned_content = clean_body_content(body_content)
            
            return {
                "dom_content": dom_content,
                "raw_content": cleaned_content,
                "images": images,
                "images_dir": images_dir
            }
            
        finally:
            driver.quit()
            logger.info("Chrome driver closed")
        
    except Exception as e:
        logger.error(f"Error scraping website: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        raise

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
