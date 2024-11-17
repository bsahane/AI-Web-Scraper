import streamlit as st
from scrape import scrape_website
from parse import analyze_content, extract_specific_info, chat_about_content
from config import initialize_session_state, render_llm_config, get_current_config

st.set_page_config(
    page_title="AI Web Scraper",
    page_icon="🌐",
    layout="wide"
)

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

def main():
    st.title("🌐 AI Web Scraper")
    
    # Initialize session states
    initialize_session_state()
    initialize_chat_state()
    
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
                
                # Store the content and update state
                st.session_state.current_website_content = content
                st.session_state.content_scraped = True
                
                # Show DOM content in expander
                with st.expander("📋 Page Structure & Content", expanded=True):
                    st.markdown(content["dom_content"])
                
                # Show raw content in expander
                with st.expander("📝 Raw Content"):
                    st.text(content["raw_content"])
                
                # Show images in expander
                if content.get("images"):
                    with st.expander("🖼️ Images", expanded=True):
                        image_cols = st.columns(3)
                        for idx, img in enumerate(content["images"]):
                            with image_cols[idx % 3]:
                                try:
                                    st.image(img["local_path"], caption=img["description"])
                                except Exception as e:
                                    st.error(f"Error displaying image: {str(e)}")
        
        # Only show tabs if content has been scraped
        if st.session_state.content_scraped:
            # Tabs for different functionalities
            tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Analysis", "🔍 Extraction"])
            
            # Chat Tab
            with tab1:
                st.subheader("Chat about the Website")
                display_chat_history()
                
                # Chat input
                if prompt := st.chat_input("Ask about the website content..."):
                    if not st.session_state.current_website_content:
                        st.error("Please scrape a website first before chatting!")
                        return
                    
                    # Add user message to chat history
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    
                    # Get AI response
                    with st.spinner("Thinking..."):
                        try:
                            response = chat_about_content(
                                st.session_state.current_website_content["raw_content"],
                                prompt,
                                llm_config,
                                st.session_state.current_website_content.get("images", [])
                            )
                            
                            # Create assistant message
                            assistant_message = {"role": "assistant", "content": response["text"]}
                            
                            # Add images if present
                            if response.get("has_images") and response.get("images"):
                                assistant_message["images"] = response["images"]
                            
                            # Add table if present
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
                    
                    # Update display without clearing
                    st.rerun()
            
            # Analysis Tab
            with tab2:
                st.subheader("General Analysis")
                if st.button("Analyze Content"):
                    with st.spinner("Analyzing content..."):
                        analysis = analyze_content(st.session_state.current_website_content["raw_content"], llm_config)
                        st.write(analysis)
            
            # Extraction Tab
            with tab3:
                st.subheader("Information Extraction")
                specific_info = st.text_input(
                    "What specific information would you like to extract?",
                    placeholder="e.g., 'Find all pricing information' or 'Extract contact details'"
                )
                if specific_info and st.button("Extract Information"):
                    with st.spinner("Extracting information..."):
                        extracted_info = extract_specific_info(st.session_state.current_website_content["raw_content"], specific_info, llm_config)
                        st.write(extracted_info)

if __name__ == "__main__":
    main()
