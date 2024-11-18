import streamlit as st
from scrape import scrape_website
from parse import analyze_content, extract_specific_info, chat_about_content, get_website_metadata, format_metadata_response
from config import initialize_session_state, render_llm_config, get_current_config
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime

st.set_page_config(
    page_title="AI Web Scraper",
    page_icon="🌐",
    layout="wide"
)

st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Vertical tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding: 12px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        background-color: #f8f9fa;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #0d6efd;
        color: white;
    }
    
    /* Content area styling */
    .stMarkdown {
        padding: 10px;
    }
    
    /* Notes editor styling */
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #dee2e6;
        padding: 12px;
        font-family: 'Monaco', monospace;
    }
    
    /* Image gallery styling */
    .stImage {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 8px;
        padding: 8px 16px;
        transition: all 0.3s ease;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 12px;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def initialize_chat_state():
    """Initialize chat-related session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_website_content' not in st.session_state:
        st.session_state.current_website_content = None
    if 'url' not in st.session_state:
        st.session_state.url = ""
    if 'content_scraped' not in st.session_state:
        st.session_state.content_scraped = False
    if 'notes' not in st.session_state:
        st.session_state.notes = {}

def display_chat_history():
    """Display the chat history"""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            # Display text content
            if "content" in message:
                st.markdown(message["content"])
            
            # Display images if present
            if "images" in message:
                cols = st.columns(min(3, len(message["images"])))
                for idx, img in enumerate(message["images"]):
                    with cols[idx % 3]:
                        try:
                            st.image(img["local_path"], caption=img["description"])
                        except Exception as e:
                            st.error(f"Error displaying image: {str(e)}")

def save_notes(url, content):
    """Save notes for the current URL"""
    st.session_state.notes[url] = content
    try:
        with open("notes.json", "w") as f:
            json.dump(st.session_state.notes, f)
    except Exception as e:
        st.error(f"Error saving notes: {str(e)}")

def load_notes():
    """Load saved notes"""
    try:
        with open("notes.json", "r") as f:
            st.session_state.notes = json.load(f)
    except FileNotFoundError:
        st.session_state.notes = {}
    except Exception as e:
        st.error(f"Error loading notes: {str(e)}")

def main():
    st.title("🌐 AI Web Scraper")
    
    # Initialize session states
    initialize_session_state()
    initialize_chat_state()
    load_notes()
    
    # Render LLM configuration in sidebar
    render_llm_config()
    
    # Get current LLM configuration
    llm_config = get_current_config()
    
    # URL input
    url = st.text_input("Enter Website URL", value=st.session_state.url, placeholder="https://example.com")
    
    # Update URL in session state
    if url != st.session_state.url:
        st.session_state.url = url
        st.session_state.content_scraped = False
        st.session_state.current_website_content = None
        st.session_state.chat_history = []
    
    if url:
        if not st.session_state.content_scraped and st.button("Scrape Website"):
            with st.spinner("Scraping website..."):
                content = scrape_website(url)
            
            if content:
                st.success("Website scraped successfully!")
                st.session_state.current_website_content = content
                st.session_state.content_scraped = True
    
    # Show content tabs if website has been scraped
    if st.session_state.content_scraped and st.session_state.current_website_content:
        # Create tabs with icons
        chat_tab, tech_tab, structure_tab, images_tab, notes_tab = st.tabs([
            "💬 Chat",
            "🔧 Technical Info",
            "📑 Page Structure",
            "🖼️ Images",
            "📒 Notes"
        ])
        
        # Chat Tab
        with chat_tab:
            st.subheader("Chat about the Website")
            display_chat_history()
            
            if prompt := st.chat_input("Ask about the website content..."):
                if not st.session_state.current_website_content:
                    st.error("Please scrape a website first before chatting!")
                    return
                
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                with st.spinner("Thinking..."):
                    try:
                        response = chat_about_content(
                            st.session_state.current_website_content["raw_content"],
                            prompt,
                            llm_config,
                            st.session_state.current_website_content.get("images", [])
                        )
                        
                        assistant_message = {"role": "assistant", "content": response["text"]}
                        
                        if response.get("has_images") and response.get("images"):
                            assistant_message["images"] = response["images"]
                        
                        if response.get("has_table") and "table" in response:
                            st.session_state.chat_history.append(assistant_message)
                            with st.chat_message("assistant"):
                                st.dataframe(
                                    response["table"],
                                    use_container_width=True,
                                    hide_index=True
                                )
                        else:
                            st.session_state.chat_history.append(assistant_message)
                        
                    except Exception as e:
                        error_msg = f"Error processing response: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_msg
                        })
                
                st.rerun()
        
        # Technical Info Tab
        with tech_tab:
            st.subheader("Website Technical Information")
            if st.button("Analyze Website"):
                with st.spinner("Analyzing website..."):
                    try:
                        metadata = get_website_metadata(url)
                        if "error" in metadata:
                            st.error(f"Error analyzing website: {metadata['error']}")
                            return
                        
                        st.markdown("## Website Technical Information")
                        
                        # Performance Metrics
                        st.markdown("### ⚡ Performance")
                        perf = metadata.get("performance", {})
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            response_time = perf.get('response_time', 'N/A')
                            if isinstance(response_time, (int, float)):
                                response_time = f"{response_time:.2f}s"
                            st.metric("Response Time", response_time)
                        with col2:
                            page_size = perf.get('page_size', 'N/A')
                            if isinstance(page_size, (int, float)):
                                page_size = f"{page_size:.1f} KB"
                            st.metric("Page Size", page_size)
                        with col3:
                            st.metric("Status Code", str(perf.get('status_code', 'N/A')))
                        
                        # Security Information
                        st.markdown("### 🔒 Security")
                        sec = metadata.get("security", {})
                        if isinstance(sec, dict) and "error" not in sec:
                            ssl_expires = sec.get('ssl_expires')
                            if isinstance(ssl_expires, datetime):
                                ssl_expires = ssl_expires.strftime('%Y-%m-%d')
                            else:
                                ssl_expires = str(ssl_expires)
                                
                            st.info(f"""
                            - SSL Issuer: {sec.get('ssl_issuer', 'N/A')}
                            - SSL Expires: {ssl_expires}
                            - SSL Version: {sec.get('ssl_version', 'N/A')}
                            """)
                        else:
                            st.warning("SSL Information not available")
                        
                        # DNS Information
                        st.markdown("### 🌐 DNS Information")
                        dns_info = metadata.get("dns", {})
                        if isinstance(dns_info, dict) and "error" not in dns_info:
                            st.info(f"""
                            - IP Address: {dns_info.get('ip_address', 'N/A')}
                            
                            **MX Records:**
                            {chr(10).join([f"- {mx}" for mx in dns_info.get('mx_records', [])[:3]])}
                            
                            **Name Servers:**
                            {chr(10).join([f"- {ns}" for ns in dns_info.get('ns_records', [])[:3]])}
                            """)
                        else:
                            st.warning("DNS Information not available")
                        
                        # Headers Analysis
                        st.markdown("### 📋 Headers")
                        headers = metadata.get("headers", {})
                        st.json({
                            "Server": headers.get("server", "N/A"),
                            "Content-Type": headers.get("content_type", "N/A"),
                            "Cache-Control": headers.get("cache_control", "N/A")
                        })
                        
                        # SEO Analysis
                        st.markdown("### 🔍 SEO Analysis")
                        seo = metadata.get("seo", {})
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            title = seo.get('title', '')
                            title_length = len(title) if title else 0
                            st.metric("Title Length", str(title_length))
                            st.metric("H1 Tags", str(seo.get('h1_count', 0)))
                        with col2:
                            st.metric("H2 Tags", str(seo.get('h2_count', 0)))
                            st.metric("H3 Tags", str(seo.get('h3_count', 0)))
                        with col3:
                            meta_desc = "Present" if seo.get('meta_description') else "Missing"
                            st.metric("Meta Description", meta_desc)
                            st.metric("Meta Tags", str(seo.get('meta_tags_count', 0)))
                        with col4:
                            st.metric("Total Images", str(seo.get('images_total', 0)))
                            st.metric("Images without Alt", str(seo.get('images_without_alt', 0)))
                        
                    except Exception as e:
                        st.error(f"Error analyzing website: {str(e)}")
        
        # Page Structure Tab
        with structure_tab:
            st.subheader("Page Structure Analysis")
            if st.button("Analyze Structure"):
                with st.spinner("Analyzing page structure..."):
                    # Fetch webpage content
                    response = requests.get(url)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Headers
                    st.markdown("### 📚 Headers")
                    headers = {
                        "h1": soup.find_all("h1"),
                        "h2": soup.find_all("h2"),
                        "h3": soup.find_all("h3")
                    }
                    for tag, elements in headers.items():
                        if elements:
                            st.markdown(f"**{tag.upper()}** ({len(elements)})")
                            for el in elements:
                                st.markdown(f"- {el.text.strip()}")
                    
                    # Links
                    st.markdown("### 🔗 Links")
                    links = soup.find_all("a")
                    st.metric("Total Links", len(links))
                    
                    # Images
                    st.markdown("### 🖼️ Images")
                    images = soup.find_all("img")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Images", len(images))
                    with col2:
                        images_no_alt = len([img for img in images if not img.get('alt')])
                        st.metric("Images without Alt", images_no_alt)
                    
                    # Forms
                    st.markdown("### 📝 Forms")
                    forms = soup.find_all("form")
                    if forms:
                        st.metric("Total Forms", len(forms))
                        for form in forms:
                            st.markdown(f"- Action: {form.get('action', 'N/A')}")
                            st.markdown(f"  Method: {form.get('method', 'N/A')}")
                    
                    # Meta Tags
                    st.markdown("### 📌 Meta Tags")
                    meta_tags = soup.find_all("meta")
                    meta_info = {}
                    for meta in meta_tags:
                        name = meta.get("name", meta.get("property", "")).lower()
                        if name:
                            meta_info[name] = meta.get("content", "N/A")
                    st.json(meta_info)
        
        # Images Tab
        with images_tab:
            if images := st.session_state.current_website_content.get("images"):
                cols = st.columns(3)
                for idx, img in enumerate(images):
                    with cols[idx % 3]:
                        try:
                            st.image(img["local_path"], caption=img["description"])
                        except Exception as e:
                            st.error(f"Error displaying image: {str(e)}")
            else:
                st.info("No images found on this page.")
        
        # Notes Tab
        with notes_tab:
            st.subheader("📝 Notes")
            current_notes = st.session_state.notes.get(url, "")
            notes_content = st.text_area(
                "Add your notes about this website (supports Markdown):",
                value=current_notes,
                height=300,
                help="You can use Markdown formatting in your notes."
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Save Notes"):
                    save_notes(url, notes_content)
                    st.success("Notes saved successfully!")
            
            if notes_content:
                st.markdown("### Preview:")
                st.markdown(notes_content)

if __name__ == "__main__":
    main()
