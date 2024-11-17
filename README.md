# AI Web Scraper

An intelligent web scraping application that combines Selenium automation with AI-powered content analysis using Ollama. This project provides a user-friendly interface built with Streamlit for scraping web content and analyzing it using AI models.

## 🚀 Features

- 🌐 Web content scraping using Selenium
- 🤖 AI-powered content analysis with Ollama
- 📊 User-friendly Streamlit interface
- 🐳 Fully containerized with Docker Compose
- 🔄 Selenium Grid integration for reliable browser automation
- 📝 Detailed content parsing and analysis
- 🛡️ Robust error handling and logging

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Scraping**: Selenium WebDriver, BeautifulSoup4
- **AI Processing**: Ollama
- **Containerization**: Docker, Docker Compose
- **Browser Automation**: Selenium Grid with Chrome
- **Language**: Python 3.11

## 📋 Prerequisites

- Docker
- Docker Compose
- Git (for cloning the repository)
- Stable internet connection

## 🔧 Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd AI-Web-Scraper-main
   ```

2. **Environment Configuration (Optional)**
   - Create a `.env` file if you want to customize default settings:
   ```env
   SELENIUM_HOST=selenium-chrome
   SELENIUM_PORT=4444
   ```

3. **Build and Start the Application**
   ```bash
   docker-compose up --build
   ```

## 🖥️ Accessing the Application

- **Main Application**: http://localhost:8501
- **Selenium Grid**: http://localhost:4444
- **VNC Viewer** (for debugging): http://localhost:7900 (password: `secret`)
- **Ollama API**: http://localhost:11434

## 🏗️ Architecture

The application is split into three main containers:

1. **Web Scraper Container**
   - Runs the main Streamlit application
   - Handles web scraping logic
   - Processes user requests

2. **Selenium Chrome Container**
   - Provides browser automation capabilities
   - Runs in headless mode
   - Managed through Selenium Grid

3. **Ollama Container**
   - Runs the AI model service
   - Handles content analysis
   - Provides AI-powered insights

## 📚 Usage Guide

1. Access the Streamlit interface at http://localhost:8501
2. Enter the URL you want to scrape
3. Select the type of analysis you want to perform
4. View the scraped content and AI analysis results

## 🛑 Troubleshooting

Common issues and solutions:

1. **Connection Issues**
   - Ensure all ports (8501, 4444, 11434) are available
   - Check if all containers are running: `docker-compose ps`

2. **Scraping Failures**
   - Verify the target URL is accessible
   - Check the Selenium Grid status at http://localhost:4444
   - Review logs: `docker-compose logs web-scraper`

3. **AI Analysis Issues**
   - Ensure Ollama container is running
   - Check Ollama logs: `docker-compose logs ollama`

## 🔒 Security Considerations

- Runs with non-root user in containers
- Disabled unnecessary Chrome extensions
- Implements certificate error handling
- Containerized environment isolation

## 🔄 Development Workflow

1. **Local Development**
   ```bash
   # Start services in development mode
   docker-compose up --build
   
   # View logs
   docker-compose logs -f
   
   # Restart specific service
   docker-compose restart web-scraper
   ```

2. **Stopping the Application**
   ```bash
   docker-compose down
   ```

## 📦 Dependencies

Key Python packages:
- selenium
- streamlit
- beautifulsoup4
- requests
- ollama-python

## ⚠️ Known Limitations

- Requires stable internet connection
- Limited to headless Chrome browsing
- Some websites may block automated access
- AI analysis dependent on Ollama model availability

## 🔜 Future Improvements

- [ ] Add more AI model options
- [ ] Implement retry mechanisms
- [ ] Enhanced error handling
- [ ] More detailed logging
- [ ] Additional scraping strategies
- [ ] User authentication
- [ ] Results caching
- [ ] Export functionality

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
