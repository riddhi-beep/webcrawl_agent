#with dynamic url search
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
        
        self.model = GenerativeModel("gemini-2.5-pro")
        self.think_and_print("Using gemini-2.5-pro model")

        
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
    
    def analyze_link_with_gemini(self, link: Dict, context: str) -> int:
        """Use Gemini to analyze how relevant a link might be"""
        try:
            prompt = f"""
            User query: {self.user_query}
            Context: I'm analyzing a website to answer the user's query.
            
            Link text: "{link['text']}"
            Link URL path: "{link['href']}"
            Current page context: {context}
            
            How likely is this link to contain information relevant to the user's query?
            Rate from 1-10 where:
            1-3: Not relevant
            4-6: Possibly relevant
            7-10: Highly relevant
            
            Respond with just the number (1-10) and a brief reason.
            """
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract the score
            score_match = re.search(r'(\d+)', response_text)
            if score_match:
                gemini_score = int(score_match.group(1))
                self.think_and_print(f"Gemini scored link '{link['text']}': {gemini_score}/10")
                return gemini_score
            
        except Exception as e:
            logging.error(f"Error in Gemini link analysis: {e}")
        
        return 0

    def prioritize_links(self, links: List[Dict], current_url: str, found_info_count: int = 0) -> List[Dict]:
        """Dynamically prioritize links based on query and current findings"""
        prioritized = []
        
        # Base keywords for different query types
        query_keywords = self.user_query.lower().split()
        
        # Dynamic keyword expansion based on what we've already found
        if found_info_count > 0:
            # If we already found some info, look for complementary information
            complementary_keywords = ['detail', 'more', 'additional', 'complete', 'full', 'specification']
        else:
            # If we haven't found much, cast a wider net
            complementary_keywords = ['overview', 'main', 'primary', 'key', 'important']
        
        page_title = ""
        try:
            soup = BeautifulSoup(self.driver.page_source if self.driver else "", 'html.parser')
            page_title = soup.title.string if soup.title else ""
        except:
            pass
        
        self.think_and_print(f"Prioritizing links from page: {page_title}")
        self.think_and_print(f"Current findings count: {found_info_count}")
        
        for link in links:
            if not link['url'] or link['url'] in self.visited_urls:
                continue
                
            score = 0
            link_text = link['text'].lower()
            href = link['href'].lower()
            
            # Dynamic scoring based on query keywords
            for keyword in query_keywords:
                if keyword in link_text:
                    score += 8  # High score for direct query keyword match in link text
                elif keyword in href:
                    score += 5  # Medium score for query keyword in URL
            
            # Complementary keywords
            for keyword in complementary_keywords:
                if keyword in link_text or keyword in href:
                    score += 2
            
            # High-value pages that often contain core information
            high_value_pages = {
                'solutions': 10, 'products': 10, 'services': 10,
                'about': 8, 'company': 8, 'contact': 7,
                'portfolio': 6, 'gallery': 6, 'testimonials': 6
            }
            
            for page, points in high_value_pages.items():
                if page in href or page in link_text:
                    score += points
            
            # Penalize less useful pages
            avoid_keywords = {
                'blog': -3, 'news': -3, 'press': -2,
                'login': -10, 'register': -10, 'cart': -10,
                'privacy': -8, 'terms': -8, 'policy': -5
            }
            
            for avoid, penalty in avoid_keywords.items():
                if avoid in href or avoid in link_text:
                    score += penalty
            
            # Use Gemini for intelligent scoring (if available)
            if score > 3:  # Only use Gemini for potentially relevant links
                gemini_score = self.analyze_link_with_gemini(link, page_title)
                if gemini_score > 0:
                    score = score + gemini_score  # Combine scores
            
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
                self.think_and_print(f"Link scored {score}: {link['text']} -> {link['url']}")
        
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
    
    def is_title_relevant(self, title: str) -> bool:
        """Check if page title is relevant to user query"""
        if not title:
            return False
            
        title_lower = title.lower()
        query_keywords = self.user_query.lower().split()
        
        # Direct keyword match in title
        direct_match = any(keyword in title_lower for keyword in query_keywords)
        
        # High-value title indicators
        valuable_titles = ['solutions', 'products', 'services', 'contact', 'about', 'company']
        title_value = any(val in title_lower for val in valuable_titles)
        
        return direct_match or title_value

    def extract_information(self, content: Dict) -> bool:
        """Extract specific information from content using Gemini"""
        if not content.get('text'):
            return False
            
        # Check if title is relevant first
        title_relevant = self.is_title_relevant(content.get('title', ''))
        
        # Always extract if title is relevant, or if content seems substantial
        should_extract = title_relevant or len(content.get('text', '')) > 500
        
        if title_relevant:
            self.think_and_print(f"🎯 Title '{content.get('title', '')}' appears relevant - extracting information")
        
        prompt = f"""
        IMPORTANT: Extract ALL information from this webpage that could answer the user's query.
        
        User Query: {self.user_query}
        
        Webpage URL: {content['url']}
        Webpage Title: {content.get('title', '')}
        Webpage Content: {content['text'][:3500]}
        
        Instructions:
        1. If the page title contains words related to the user's query, this page is HIGHLY RELEVANT
        2. Extract EVERYTHING that could answer the user's query, including:
           - Complete product/service descriptions
           - All contact information (phone, email, address)
           - Location and operational details
           - Export/international information
           - Certificates, quality standards, compliance info
           - Company information and capabilities
           - Any other details that relate to the query
        
        3. Be comprehensive - don't just extract a summary, extract the actual details
        4. If this appears to be a solutions/products/services page, extract ALL offerings mentioned
        5. Preserve specific names, numbers, addresses, and detailed descriptions
        
        Format your response with clear sections if multiple types of information are found.
        Only say "No relevant information found" if the page truly contains nothing related to the query.
        """
        
        try:
            response = self.model.generate_content(prompt)
            extracted_info = response.text.strip()
            
            # More lenient check - extract unless explicitly no information
            if extracted_info and len(extracted_info) > 20:
                # Check if it's actually relevant
                query_words = self.user_query.lower().split()
                info_lower = extracted_info.lower()
                
                # If title was relevant OR content contains query keywords, keep it
                if title_relevant or any(word in info_lower for word in query_words):
                    self.relevant_info.append({
                        'url': content['url'],
                        'title': content.get('title', ''),
                        'information': extracted_info,
                        'title_relevant': title_relevant
                    })
                    self.think_and_print(f"✅ Extracted information from {content['url']} (Title relevant: {title_relevant})")
                    return True
                else:
                    self.think_and_print(f"❌ Extracted info not relevant to query from {content['url']}")
            else:
                self.think_and_print(f"❌ No substantial information extracted from {content['url']}")
                
        except Exception as e:
            logging.error(f"Error extracting information from {content['url']}: {e}")
            # Fallback: simple text analysis
            return self.fallback_extraction(content)
        
        return False
    
    def fallback_extraction(self, content: Dict) -> bool:
        """Fallback extraction when Gemini fails"""
        text = content['text']
        query_keywords = self.user_query.lower().split()
        
        # Look for sentences containing query keywords
        sentences = text.split('.')
        relevant_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in query_keywords):
                relevant_sentences.append(sentence.strip())
        
        # Also check if title suggests this page should have relevant content
        title_relevant = self.is_title_relevant(content.get('title', ''))
        
        if relevant_sentences or title_relevant:
            if relevant_sentences:
                info = " ".join(relevant_sentences[:5])  # Take first 5 relevant sentences
            else:
                # If title is relevant but no specific sentences found, take beginning of content
                info = text[:1000] + "..." if len(text) > 1000 else text
            
            self.relevant_info.append({
                'url': content['url'],
                'title': content.get('title', ''),
                'information': info,
                'title_relevant': title_relevant
            })
            self.think_and_print(f"✅ Used fallback extraction for {content['url']}")
            return True
        
        return False
    
    def analyze_website(self, start_url: str, user_query: str):
        """Main analysis function with dynamic link prioritization"""
        self.user_query = user_query
        self.think_and_print(f"Starting analysis for query: '{user_query}'")
        self.think_and_print(f"Starting from homepage: {start_url}")
        
        # Start with homepage
        homepage_content = self.crawl_page(start_url)
        
        if not homepage_content:
            self.think_and_print("Failed to crawl homepage")
            return
        
        self.think_and_print(f"Successfully crawled homepage. Text length: {len(homepage_content.get('text', ''))}")
        
        # Extract information from homepage
        homepage_success = self.extract_information(homepage_content)
        current_findings = len(self.relevant_info)
        
        # Get and prioritize links dynamically
        prioritized_links = self.prioritize_links(homepage_content['links'], start_url, current_findings)
        
        self.think_and_print(f"Found {len(prioritized_links)} potentially relevant links to explore")
        
        # Crawl prioritized pages with adaptive strategy
        links_to_crawl = min(8, len(prioritized_links))  # Increased from 5 to 8
        
        for i, link in enumerate(prioritized_links[:links_to_crawl]):
            self.think_and_print(f"Exploring link {i+1}/{links_to_crawl}: {link['url']}")
            self.think_and_print(f"Link text: '{link['text']}'")
            
            page_content = self.crawl_page(link['url'], depth=1)
            if page_content:
                # Check if title suggests this is exactly what we're looking for
                title = page_content.get('title', '')
                if title:
                    self.think_and_print(f"Page title: '{title}'")
                
                # Extract information - this will now be more aggressive
                extraction_success = self.extract_information(page_content)
                
                if extraction_success:
                    self.think_and_print(f"Successfully extracted information from {link['url']}")
                    
                    # If this page had relevant title and we extracted info, process images too
                    if self.is_title_relevant(title):
                        self.process_images(page_content)
                        
                        # If this was a high-value page (like solutions), look for more links on this page
                        if any(keyword in title.lower() for keyword in ['solution', 'product', 'service']):
                            self.think_and_print(f"Found high-value page '{title}' - checking for sub-pages")
                            sub_links = self.prioritize_links(page_content['links'], link['url'], len(self.relevant_info))
                            
                            # Crawl top 2 sub-links from this valuable page
                            for j, sub_link in enumerate(sub_links[:2]):
                                if sub_link['url'] not in self.visited_urls:
                                    self.think_and_print(f"  Exploring sub-link {j+1}: {sub_link['url']}")
                                    sub_content = self.crawl_page(sub_link['url'], depth=2)
                                    if sub_content:
                                        self.extract_information(sub_content)
                else:
                    self.think_and_print(f"No relevant information extracted from {link['url']}")
            
            # Add delay to be respectful
            time.sleep(1)
        
        self.think_and_print(f"Analysis complete. Found {len(self.relevant_info)} sources of information.")
    
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
        """Generate final comprehensive report ensuring all extracted info is used"""
        if not self.relevant_info:
            return "No relevant information found for the given query. The website may not contain the information you're looking for, or it might be behind authentication/JavaScript that couldn't be accessed."
        
        self.think_and_print(f"Generating report from {len(self.relevant_info)} sources")
        
        # Organize information by priority (title-relevant first)
        title_relevant_info = [info for info in self.relevant_info if info.get('title_relevant', False)]
        other_info = [info for info in self.relevant_info if not info.get('title_relevant', False)]
        
        self.think_and_print(f"Title-relevant sources: {len(title_relevant_info)}")
        self.think_and_print(f"Other relevant sources: {len(other_info)}")
        
        # Create comprehensive information summary
        all_info_text = "PRIORITY INFORMATION (from pages with relevant titles):\n\n"
        
        for i, item in enumerate(title_relevant_info, 1):
            all_info_text += f"{i}. Source: {item['url']}\n"
            all_info_text += f"   Page Title: {item['title']}\n"
            all_info_text += f"   Information: {item['information']}\n\n"
        
        if other_info:
            all_info_text += "ADDITIONAL INFORMATION:\n\n"
            for i, item in enumerate(other_info, 1):
                all_info_text += f"{i}. Source: {item['url']}\n"
                all_info_text += f"   Page Title: {item['title']}\n"
                all_info_text += f"   Information: {item['information']}\n\n"
        
        prompt = f"""
        User Query: "{self.user_query}"
        
        I have collected comprehensive information from a website analysis:
        
        {all_info_text}
        
        IMPORTANT INSTRUCTIONS:
        1. Create a complete answer to the user's query using ALL the relevant information provided above
        2. Prioritize information from pages with relevant titles (marked as "PRIORITY INFORMATION")
        3. Be comprehensive - include specific details, names, descriptions, contact info, etc.
        4. Organize the response logically to directly address what the user asked
        5. If multiple sources provide the same type of information, combine and present it clearly
        6. DO NOT ignore any relevant details - the user wants a thorough answer
        7. If solutions/products/services were found, list them specifically
        
        Structure your response to be helpful and actionable for the user.
        """
        
        try:
            response = self.model.generate_content(prompt)
            final_report = response.text
            
            # Ensure the report actually uses the extracted information
            if len(final_report) < 200 and len(all_info_text) > 500:
                self.think_and_print("Report seems too short given available info - using fallback")
                return self.create_fallback_report()
            
            return final_report
            
        except Exception as e:
            logging.error(f"Error generating final report: {e}")
            return self.create_fallback_report()
    
    def create_fallback_report(self) -> str:
        """Create a structured report when Gemini fails"""
        if not self.relevant_info:
            return "No relevant information found."
        
        report = f"Based on analysis of the website, here's what I found regarding: {self.user_query}\n\n"
        
        # Group by title-relevant vs other
        title_relevant = [info for info in self.relevant_info if info.get('title_relevant', False)]
        other_relevant = [info for info in self.relevant_info if not info.get('title_relevant', False)]
        
        if title_relevant:
            report += "KEY FINDINGS (from directly relevant pages):\n\n"
            for i, item in enumerate(title_relevant, 1):
                report += f"{i}. From: {item['title']} ({item['url']})\n"
                report += f"   {item['information']}\n\n"
        
        if other_relevant:
            report += "ADDITIONAL INFORMATION:\n\n"
            for i, item in enumerate(other_relevant, 1):
                report += f"{i}. From: {item['title']} ({item['url']})\n"
                report += f"   {item['information']}\n\n"
        
        return report
    
    def cleanup(self):
        """Clean up resources"""
        if self.driver:
            self.driver.quit()

# Main execution function
def main():
    # Your Vertex AI credentials
    PROJECT_ID = "wired-name-455213-q8"
    LOCATION = "us-central1"  # Changed to a more common location
    CREDENTIALS_PATH = "/Users/riddhi/Desktop/crawl_agent/wired-name-455213-q8-41933c90f1bf.json"
    
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