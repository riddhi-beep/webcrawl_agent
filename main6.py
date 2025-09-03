#deepseek + claude with static url search
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import re
from typing import List, Dict, Optional
import time
from PIL import Image
import io
import base64
import pytesseract
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import logging
import google.auth
from google.oauth2 import service_account

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WebsiteAnalyzerAgent:
    def __init__(self, project_id: str, location: str = "us-central1", credentials_path: str = None):
        # Configure authentication
        if credentials_path:
            # Use service account credentials from file
            self.credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
        else:
            # Try to use default credentials (for environments like Google Cloud)
            self.credentials, _ = google.auth.default()
        
        # Initialize Vertex AI with explicit credentials
        vertexai.init(
            project=project_id, 
            location=location,
            credentials=self.credentials
        )
        
        # Use correct model name - try gemini-1.5-pro first, fallback to gemini-pro
        # try:
        self.model = GenerativeModel("gemini-2.5-pro")
        self.think_and_print("Using gemini-2.5-pro model")
        # except Exception as e:
        #     try:
        #         self.model = GenerativeModel("gemini-pro")
        #         self.think_and_print("Using gemini-pro model")
        #     except Exception as e2:
        #         logging.error(f"Failed to initialize model: {e2}")
        #         raise e2
        
        # Initialize Selenium WebDriver (for JavaScript-heavy sites)
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        
        try:
            self.driver = webdriver.Chrome(options=options)
        except Exception as e:
            logging.error(f"Failed to initialize Chrome driver: {e}")
            # Fallback to requests-only mode
            self.driver = None
        
        self.visited_urls = set()
        self.max_depth = 2  # Reduced depth to avoid too many requests
        self.relevant_info = []
        self.user_query = ""
        
    def think_and_print(self, message: str):
        """Print agent's thinking process"""
        print(f"🤔 AGENT THINKING: {message}")
        logging.info(message)
    
    def extract_text_from_image(self, image_url: str) -> str:
        """Extract text from images using OCR"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(image_url, headers=headers, timeout=10)
            if response.status_code == 200:
                # Use pytesseract for OCR
                image = Image.open(io.BytesIO(response.content))
                text = pytesseract.image_to_string(image)
                return text.strip()
        except Exception as e:
            logging.error(f"Error in OCR for {image_url}: {e}")
            return ""
        return ""
    
    def analyze_image_with_gemini(self, image_url: str, context: str) -> str:
        """Use Gemini to analyze image content"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(image_url, headers=headers, timeout=10)
            if response.status_code == 200:
                # Create image part for Gemini
                image_part = Part.from_data(
                    data=response.content,
                    mime_type=response.headers.get('content-type', 'image/jpeg')
                )
                
                prompt = f"""
                Analyze this image in the context of: {context}
                User query: {self.user_query}
                
                What information does this image contain that might be relevant to the user's query?
                If it contains text, describe the text content.
                If it shows products, locations, certificates, or other relevant information, describe it.
                Be concise but specific.
                """
                
                response = self.model.generate_content([prompt, image_part])
                return response.text
        except Exception as e:
            logging.error(f"Error analyzing image with Gemini: {e}")
            return ""
        return ""
    
    def crawl_page(self, url: str, depth: int = 0) -> Dict:
        """Crawl a webpage and extract content"""
        if depth > self.max_depth or url in self.visited_urls:
            return {}
        
        self.visited_urls.add(url)
        self.think_and_print(f"Crawling: {url} (depth: {depth})")
        
        try:
            if self.driver:
                # Use Selenium for JavaScript rendering
                self.driver.get(url)
                time.sleep(2)  # Wait for page to load
                page_source = self.driver.page_source
            else:
                # Fallback to requests
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                page_source = response.text
            
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text content
            text_content = soup.get_text(separator=' ', strip=True)
            text_content = re.sub(r'\s+', ' ', text_content)
            
            # Extract links
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('#') or href.startswith('javascript:'):
                    continue
                    
                absolute_url = urljoin(url, href)
                if self.is_same_domain(absolute_url, url):
                    links.append({
                        'url': absolute_url,
                        'text': link.get_text(strip=True),
                        'href': href
                    })
            
            # Extract images
            images = []
            for img in soup.find_all('img', src=True):
                img_url = urljoin(url, img['src'])
                alt_text = img.get('alt', '')
                images.append({
                    'url': img_url,
                    'alt': alt_text
                })
            
            return {
                'url': url,
                'text': text_content,
                'links': links,
                'images': images,
                'title': soup.title.string if soup.title else ''
            }
            
        except Exception as e:
            logging.error(f"Error crawling {url}: {e}")
            return {}
    
    def is_same_domain(self, url1: str, url2: str) -> bool:
        """Check if two URLs belong to the same domain"""
        try:
            domain1 = urlparse(url1).netloc.lower()
            domain2 = urlparse(url2).netloc.lower()
            return domain1 == domain2
        except:
            return False
    
    def analyze_content_relevance(self, content: Dict) -> bool:
        """Use Gemini to analyze if content is relevant to user query"""
        if not content.get('text'):
            return False
            
        # Simple keyword-based relevance check as fallback
        query_keywords = self.user_query.lower().split()
        content_text = content['text'].lower()
        
        keyword_match = any(keyword in content_text for keyword in query_keywords)
        
        # Try Gemini analysis
        try:
            prompt = f"""
            User query: {self.user_query}
            Webpage content from {content['url']}:
            Title: {content.get('title', '')}
            Content preview: {content['text'][:1500]}
            
            Is this content relevant to answering the user's query? 
            Consider if it contains information about what the user is asking for.
            
            Respond with only "YES" or "NO" followed by a brief reason.
            """
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            self.think_and_print(f"Relevance analysis for {content['url']}: {response_text}")
            
            is_relevant = "YES" in response_text.upper()
            return is_relevant or keyword_match  # Use either Gemini or keyword match
            
        except Exception as e:
            logging.error(f"Error in relevance analysis: {e}")
            # Fallback to keyword matching
            self.think_and_print(f"Using keyword matching for relevance. Match found: {keyword_match}")
            return keyword_match
    
    def prioritize_links(self, links: List[Dict], current_url: str) -> List[Dict]:
        """Prioritize links based on likelihood of containing relevant info"""
        prioritized = []
        
        # Keywords that might indicate relevant pages
        relevant_keywords = {
            'products': ['product', 'manufacture', 'offerings', 'solutions', 'services', 'catalog'],
            'locations': ['location', 'office', 'global', 'country', 'region', 'contact', 'address'],
            'export': ['export', 'international', 'global', 'countries', 'worldwide'],
            'contact': ['contact', 'reach', 'phone', 'email', 'address', 'info'],
            'certificates': ['certificate', 'certification', 'quality', 'standard', 'iso', 'compliance'],
            'clients': ['client', 'customer', 'testimonial', 'case study', 'portfolio', 'gallery'],
            'about': ['about', 'company', 'history', 'mission', 'vision', 'profile']
        }
        
        query_type = self.determine_query_type(self.user_query)
        self.think_and_print(f"Detected query type: {query_type}")
        
        # Get relevant keywords for this query type
        relevant_keywords_list = relevant_keywords.get(query_type, [])
        # Also include general keywords
        relevant_keywords_list.extend(['about', 'company', 'info'])
        
        for link in links:
            if not link['url'] or link['url'] in self.visited_urls:
                continue
                
            score = 0
            link_text = link['text'].lower()
            href = link['href'].lower()
            
            # Score based on keywords
            for keyword in relevant_keywords_list:
                if keyword in link_text or keyword in href:
                    score += 3
            
            # Prioritize common relevant pages
            common_pages = ['about', 'contact', 'products', 'services', 'locations', 'global', 'company']
            for page in common_pages:
                if page in href:
                    score += 2
            
            # Avoid certain types of links
            avoid_keywords = ['blog', 'news', 'login', 'register', 'cart', 'checkout', 'privacy', 'terms']
            for avoid in avoid_keywords:
                if avoid in href:
                    score -= 5
            
            if score > 0:
                prioritized.append((score, link))
        
        # Sort by score (highest first) and remove duplicates
        prioritized.sort(key=lambda x: x[0], reverse=True)
        unique_links = []
        seen_urls = set()
        
        for score, link in prioritized:
            if link['url'] not in seen_urls:
                unique_links.append(link)
                seen_urls.add(link['url'])
        
        self.think_and_print(f"Prioritized {len(unique_links)} unique links")
        return unique_links
    
    def determine_query_type(self, query: str) -> str:
        """Determine what type of information the user is looking for"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['product', 'manufacture', 'make', 'offer', 'sell']):
            return 'products'
        elif any(word in query_lower for word in ['location', 'operate', 'office', 'country', 'where']):
            return 'locations'
        elif any(word in query_lower for word in ['export', 'ship to', 'international', 'global']):
            return 'export'
        elif any(word in query_lower for word in ['contact', 'phone', 'email', 'address', 'reach']):
            return 'contact'
        elif any(word in query_lower for word in ['certificate', 'certification', 'quality', 'standard']):
            return 'certificates'
        elif any(word in query_lower for word in ['client', 'customer', 'testimonial', 'review']):
            return 'clients'
        else:
            return 'about'
    
    def extract_information(self, content: Dict):
        """Extract specific information from content using Gemini"""
        if not content.get('text'):
            return
            
        prompt = f"""
        Extract specific information from this webpage content that answers the user's query.
        
        User Query: {self.user_query}
        
        Webpage URL: {content['url']}
        Webpage Title: {content.get('title', '')}
        Webpage Content: {content['text'][:2500]}
        
        Extract only the information that directly answers the user's query.
        Be specific and include relevant details like:
        - Product names and descriptions
        - Contact information (phone, email, address)
        - Location details
        - Export information
        - Certificates or quality standards
        - Any other relevant facts
        
        Format the response clearly and concisely. If no relevant information is found, say "No relevant information found."
        """
        
        try:
            response = self.model.generate_content(prompt)
            extracted_info = response.text.strip()
            
            if extracted_info and "no relevant information" not in extracted_info.lower():
                self.relevant_info.append({
                    'url': content['url'],
                    'title': content.get('title', ''),
                    'information': extracted_info
                })
                self.think_and_print(f"✅ Extracted relevant information from {content['url']}")
            else:
                self.think_and_print(f"❌ No relevant information found on {content['url']}")
                
        except Exception as e:
            logging.error(f"Error extracting information from {content['url']}: {e}")
            # Fallback: simple text analysis
            self.fallback_extraction(content)
    
    def fallback_extraction(self, content: Dict):
        """Fallback extraction when Gemini fails"""
        text = content['text'].lower()
        query_keywords = self.user_query.lower().split()
        
        # Look for sentences containing query keywords
        sentences = content['text'].split('.')
        relevant_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in query_keywords):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            info = " ".join(relevant_sentences[:3])  # Take first 3 relevant sentences
            self.relevant_info.append({
                'url': content['url'],
                'title': content.get('title', ''),
                'information': info
            })
            self.think_and_print(f"✅ Used fallback extraction for {content['url']}")
    
    def analyze_website(self, start_url: str, user_query: str):
        """Main analysis function"""
        self.user_query = user_query
        self.think_and_print(f"Starting analysis for query: '{user_query}'")
        self.think_and_print(f"Starting from homepage: {start_url}")
        
        # Start with homepage
        homepage_content = self.crawl_page(start_url)
        
        if not homepage_content:
            self.think_and_print("❌ Failed to crawl homepage")
            return
        
        self.think_and_print(f"✅ Successfully crawled homepage. Text length: {len(homepage_content.get('text', ''))}")
        
        # Always extract information from homepage
        self.extract_information(homepage_content)
        
        # Get and prioritize links
        prioritized_links = self.prioritize_links(homepage_content['links'], start_url)
        
        self.think_and_print(f"Found {len(prioritized_links)} potentially relevant links to explore")
        
        # Crawl prioritized pages (limit to top 5 to avoid too many requests)
        for i, link in enumerate(prioritized_links[:5]):
            self.think_and_print(f"📄 Exploring link {i+1}: {link['url']}")
            
            page_content = self.crawl_page(link['url'], depth=1)
            if page_content:
                # Always try to extract information, let the extraction function decide relevance
                self.extract_information(page_content)
                
                # Only process images if we found relevant content
                if any(info['url'] == page_content['url'] for info in self.relevant_info):
                    self.process_images(page_content)
            
            # Add small delay to be respectful
            time.sleep(1)
    
    def process_images(self, content: Dict):
        """Process images on a page"""
        if not content.get('images'):
            return
            
        self.think_and_print(f"🖼️ Processing {len(content['images'])} images from {content['url']}")
        
        # Limit to first 3 images to avoid too many requests
        for image in content['images'][:3]:
            try:
                # Skip if image URL is incomplete
                if not image['url'] or image['url'].startswith('data:'):
                    continue
                
                self.think_and_print(f"Analyzing image: {image['url']}")
                
                # Try OCR first (faster)
                ocr_text = self.extract_text_from_image(image['url'])
                if ocr_text and len(ocr_text) > 10:  # Only if substantial text found
                    query_keywords = self.user_query.lower().split()
                    if any(keyword in ocr_text.lower() for keyword in query_keywords):
                        self.relevant_info.append({
                            'url': image['url'],
                            'title': f"Image from {content['url']}",
                            'information': f"Image text (OCR): {ocr_text}"
                        })
                        self.think_and_print(f"✅ Found relevant text in image via OCR")
                
            except Exception as e:
                logging.error(f"Error processing image {image['url']}: {e}")
    
    def generate_final_report(self) -> str:
        """Generate final comprehensive report"""
        if not self.relevant_info:
            return "❌ No relevant information found for the given query. The website may not contain the information you're looking for, or it might be behind authentication/JavaScript that couldn't be accessed."
        
        self.think_and_print(f"📊 Generating report from {len(self.relevant_info)} sources")
        
        # Create a summary of all found information
        all_info = []
        for item in self.relevant_info:
            source_info = f"Source: {item['url']}\n"
            if item.get('title'):
                source_info += f"Page: {item['title']}\n"
            source_info += f"Information: {item['information']}\n"
            all_info.append(source_info)
        
        prompt = f"""
        User Query: {self.user_query}
        
        I have gathered the following information from the website:
        
        {chr(10).join(all_info)}
        
        Please create a comprehensive, well-structured answer to the user's query.
        Organize the information clearly and remove any duplicates.
        Provide specific details and be as helpful as possible.
        If you found contact information, locations, products, or other specific details, include them.
        Structure your response in a logical way that directly addresses the user's question.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Error generating final report: {e}")
            # Fallback: return raw information
            fallback_report = f"Found information from {len(self.relevant_info)} sources:\n\n"
            for item in self.relevant_info:
                fallback_report += f"From {item['url']}:\n{item['information']}\n\n"
            return fallback_report
    
    def cleanup(self):
        """Clean up resources"""
        if self.driver:
            self.driver.quit()

# Main execution function
def main():
    # Your Vertex AI credentials
    PROJECT_ID = "wired-name-455213-q8"
    LOCATION = "us-west1"  # Changed to a more common location
    CREDENTIALS_PATH = "/Users/riddhi/Desktop/crawl/wired-name-455213-q8-41933c90f1bf.json"
    
    # Get user input
    START_URL = input("Enter the website URL to analyze: ").strip()
    USER_QUERY = input("What information would you like to know from the website? ").strip()
    
    # Add https:// if not present
    if not START_URL.startswith(('http://', 'https://')):
        START_URL = 'https://' + START_URL
    
    # Initialize agent with your credentials
    agent = WebsiteAnalyzerAgent(
        project_id=PROJECT_ID,
        location=LOCATION,
        credentials_path=CREDENTIALS_PATH
    )
    
    try:
        # Perform analysis
        agent.analyze_website(START_URL, USER_QUERY)
        
        # Generate and display final report
        print("\n" + "="*70)
        print("🔍 WEBSITE ANALYSIS REPORT")
        print("="*70)
        print(f"Query: {USER_QUERY}")
        print(f"Website: {START_URL}")
        print("-"*70)
        
        report = agent.generate_final_report()
        print(report)
        
        print("\n" + "="*70)
        print(f"📈 Analysis completed. Visited {len(agent.visited_urls)} pages.")
        print("="*70)
        
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        print(f"❌ An error occurred: {e}")
        print("Please check your credentials and try again.")
    
    finally:
        agent.cleanup()

if __name__ == "__main__":
    main()