import streamlit as st
from scrape import scrape_website
from parse import analyze_content, extract_specific_info
from config import initialize_session_state, render_llm_config, get_current_config

st.set_page_config(
    page_title="AI Web Scraper",
    page_icon="🌐",
    layout="wide"
)

def main():
    st.title("🌐 AI Web Scraper")
    
    # Initialize session state for LLM configuration
    initialize_session_state()
    
    # Render LLM configuration in sidebar
    render_llm_config()
    
    # Get current LLM configuration
    llm_config = get_current_config()
    
    # Main content
    url = st.text_input("Enter Website URL")
    
    if url:
        try:
            with st.spinner("Scraping website..."):
                content = scrape_website(url)
            
            if content:
                st.success("Website scraped successfully!")
                
                # Show raw content in expander
                with st.expander("View Raw Content"):
                    st.text(content)
                
                # Analysis options
                analysis_type = st.radio(
                    "Select Analysis Type",
                    ["General Analysis", "Specific Information Extraction"]
                )
                
                if analysis_type == "General Analysis":
                    if st.button("Analyze Content"):
                        with st.spinner("Analyzing content..."):
                            analysis = analyze_content(content, llm_config)
                            st.write(analysis)
                
                else:
                    specific_info = st.text_input(
                        "What specific information would you like to extract?",
                        placeholder="e.g., 'Find all pricing information' or 'Extract contact details'"
                    )
                    if specific_info and st.button("Extract Information"):
                        with st.spinner("Extracting information..."):
                            extracted_info = extract_specific_info(content, specific_info, llm_config)
                            st.write(extracted_info)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
