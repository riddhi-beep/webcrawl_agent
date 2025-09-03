#all crawling
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import re
import time
from PIL import Image
import io
import logging
import google.auth
from google.oauth2 import service_account
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# ✅ ADD MISSING pytesseract IMPORT
import pytesseract  # Make sure `pytesseract` and `tesseract` are installed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SmartWebsiteAnalyzerAgent:
    def __init__(self, project_id: str, location: str = "us-central1", credentials_path: str = None):
        # Authentication
        if credentials_path:
            self.credentials = service_account.Credentials.from_service_account_file(credentials_path)
        else:
            self.credentials, _ = google.auth.default()
        
        vertexai.init(project=project_id, location=location, credentials=self.credentials)
        
        try:
            self.model = GenerativeModel("gemini-2.5-pro")
            logging.info("Using gemini-2.5-pro model")
        except Exception as e:
            logging.warning(f"Gemini 1.5 Pro not available: {e}, falling back to gemini-pro")
            self.model = GenerativeModel("gemini-pro")

        # WebDriver setup
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')

        try:
            self.driver = webdriver.Chrome(options=options)
        except Exception as e:
            logging.error(f"Chrome driver failed to initialize: {e}")
            self.driver = None

        self.base_domain = ""
        self.all_urls = set()
        self.visited_urls = set()
        self.user_query = ""
        self.relevant_info = []
        self.max_depth = 3  # Increased depth for thoroughness

    def think_and_print(self, message: str):
        print(f"🧠 {message}")
        logging.info(message)

    def is_internal_url(self, url: str) -> bool:
        """Check if URL belongs to the same domain"""
        try:
            return urlparse(url).netloc.lower() == self.base_domain
        except:
            return False

    def crawl_site(self, start_url: str):
        """Crawl the entire site (within domain) and collect all internal URLs"""
        self.base_domain = urlparse(start_url).netloc.lower()
        queue = [(start_url, 0)]  # (url, depth)

        self.think_and_print(f"🌐 Starting full-site crawl from: {start_url}")

        while queue:
            url, depth = queue.pop(0)
            
            if depth > self.max_depth or url in self.all_urls:
                continue

            self.all_urls.add(url)
            self.think_and_print(f"🔗 Discovered: {url}")

            try:
                if self.driver:
                    self.driver.get(url)
                    time.sleep(2)  # Allow JS to load
                    page_source = self.driver.page_source
                else:
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                    response = requests.get(url, headers=headers, timeout=15)
                    page_source = response.text

                soup = BeautifulSoup(page_source, 'html.parser')
                for tag in soup(["script", "style"]):
                    tag.decompose()

                # Extract all links
                for link in soup.find_all('a', href=True):
                    href = link['href'].strip()
                    if href.startswith(('javascript:', 'mailto:', 'tel:', '#', 'tel:', 'whatsapp:')):
                        continue
                    absolute_url = urljoin(url, href)
                    if self.is_internal_url(absolute_url):
                        queue.append((absolute_url, depth + 1))

            except Exception as e:
                logging.error(f"Failed to crawl {url}: {e}")

        self.think_and_print(f"✅ Crawl complete. Found {len(self.all_urls)} internal URLs.")

    def score_url(self, url: str, title: str = "", link_text: str = "") -> int:
        """Score a URL for relevance to the user query"""
        score = 0
        query_lower = self.user_query.lower()
        url_lower = url.lower()
        title_lower = title.lower() if title else ""
        text_lower = link_text.lower() if link_text else ""

        # Keyword matches
        keywords = self.user_query.lower().split()
        for kw in keywords:
            if kw in title_lower:
                score += 20
            elif kw in text_lower:
                score += 15
            elif kw in url_lower:
                score += 10

        # High-value pages
        high_value = ['product', 'service', 'solution', 'contact', 'about', 'company', 'location', 'export', 'certificate', 'client', 'testimonial', 'portfolio']
        for word in high_value:
            if word in title_lower or word in url_lower or word in text_lower:
                score += 8

        # Penalize low-value pages
        low_value = ['blog', 'news', 'press', 'login', 'register', 'cart', 'checkout', 'privacy', 'terms', 'policy', 'help']
        for word in low_value:
            if word in url_lower or word in text_lower:
                score -= 15

        # ✅ Use Gemini for intelligent scoring if promising
        if score > 10:
            try:
                prompt = f"""
                User query: {self.user_query}
                Candidate URL: {url}
                Page title: {title}
                Link text: {link_text}

                On a scale of 1-10, how likely is this page to contain information that answers the user's query?
                Respond with only the number.
                """
                response = self.model.generate_content(prompt)
                gemini_score = re.search(r'\d+', response.text.strip())
                if gemini_score:
                    score += int(gemini_score.group()) * 3  # Strong AI boost
            except Exception as e:
                logging.warning(f"Gemini scoring failed for {url}: {e}")

        return max(score, 0)

    def rank_urls(self) -> list:
        """Rank all discovered URLs by relevance score"""
        ranked = []

        for url in self.all_urls:
            title, link_text = "", ""

            # Try to get title from direct crawl or URL
            try:
                if self.driver:
                    self.driver.get(url)
                    time.sleep(1)
                    soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                    title = soup.title.string if soup.title else ""
            except Exception as e:
                logging.debug(f"Could not get title for {url}: {e}")

            score = self.score_url(url, title, link_text)
            ranked.append({'url': url, 'score': score, 'title': title})

        # Sort by score descending
        ranked.sort(key=lambda x: x['score'], reverse=True)
        return ranked

    def extract_text_from_image(self, image_url: str) -> str:
        """Use OCR to extract text from image"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(image_url, headers=headers, timeout=15)
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                return pytesseract.image_to_string(image).strip()
        except Exception as e:
            logging.error(f"OCR failed for {image_url}: {e}")
        return ""

    def process_images(self, content: dict):
        """Process ALL images on the page (no limit)"""
        if not content.get('images'):
            return

        self.think_and_print(f"🖼️ Processing {len(content['images'])} images from {content['url']}")

        for img in content['images']:
            img_url = img['url']
            if not img_url or img_url.startswith('data:') or img_url.startswith('blob:'):
                continue

            try:
                self.think_and_print(f"🔍 Analyzing image: {img_url}")

                # OCR first
                ocr_text = self.extract_text_from_image(img_url)
                if ocr_text and len(ocr_text) > 5:
                    if self.user_query.lower() in ocr_text.lower():
                        self.relevant_info.append({
                            'url': img_url,
                            'title': f"Text from image on {content['url']}",
                            'information': f"📄 OCR Extracted Text: {ocr_text}"
                        })
                        self.think_and_print(f"✅ Found relevant text in image via OCR: {img_url}")

                # Optional: Use Gemini Vision for deeper analysis
                # Uncomment if you want to use Gemini for image understanding
                
                try:
                    image_response = requests.get(img_url, headers=headers, timeout=15)
                    if image_response.status_code == 200:
                        image_part = Part.from_data(
                            data=image_response.content,
                            mime_type=image_response.headers.get('content-type', 'image/jpeg')
                        )
                        prompt = f"Describe this image in context of: {self.user_query}"
                        response = self.model.generate_content([prompt, image_part])
                        gemini_desc = response.text.strip()
                        if gemini_desc and "no relevant" not in gemini_desc.lower():
                            self.relevant_info.append({
                                'url': img_url,
                                'title': f"Image analysis from {content['url']}",
                                'information': f"👁️ Gemini Vision: {gemini_desc}"
                            })
                except Exception as ie:
                    logging.warning(f"Gemini image analysis failed: {ie}")
                

            except Exception as e:
                logging.error(f"Error processing image {img_url}: {e}")

    def extract_information(self, content: dict) -> bool:
        """Use Gemini to extract all relevant info from the page"""
        if not content.get('text'):
            return False

        prompt = f"""
        USER QUERY: "{self.user_query}"
        
        FULL PAGE CONTENT from {content['url']}:
        TITLE: {content.get('title', 'N/A')}
        CONTENT: {content['text']}  # No truncation — full content passed

        INSTRUCTIONS:
        - Extract EVERY piece of information that could answer the user's query.
        - Include product names, contact details, locations, certifications, client names, export regions, etc.
        - Be extremely thorough. Do not summarize unless necessary.
        - If the title or URL suggests relevance, assume the page is important.
        - Even partial matches should be included.
        - Respond with raw facts, not commentary.
        - If nothing relevant, say "No relevant information".
        """

        try:
            response = self.model.generate_content(prompt)
            extracted = response.text.strip()
            if extracted and "no relevant information" not in extracted.lower():
                self.relevant_info.append({
                    'url': content['url'],
                    'title': content.get('title', ''),
                    'information': extracted
                })
                self.think_and_print(f"✅ Extracted information from: {content['url']}")
                return True
        except Exception as e:
            logging.error(f"Extraction failed for {content['url']}: {e}")
        return False

    def crawl_and_analyze_page(self, url: str) -> bool:
        """Crawl and analyze a single page"""
        if url in self.visited_urls:
            return False
        self.visited_urls.add(url)

        try:
            if self.driver:
                self.driver.get(url)
                time.sleep(2)
                page_source = self.driver.page_source
            else:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.get(url, headers=headers, timeout=15)
                page_source = response.text

            soup = BeautifulSoup(page_source, 'html.parser')
            for tag in soup(["script", "style"]):
                tag.decompose()

            text = soup.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace

            # Extract all images
            images = [
                {'url': urljoin(url, img['src']), 'alt': img.get('alt', '')}
                for img in soup.find_all('img', src=True)
            ]

            content = {
                'url': url,
                'text': text,
                'title': soup.title.string if soup.title else '',
                'images': images
            }

            found = self.extract_information(content)
            if found:
                self.process_images(content)
            return found

        except Exception as e:
            logging.error(f"Failed to analyze {url}: {e}")
            return False

    def analyze_website(self, start_url: str, user_query: str):
        """Main analysis: crawl → rank → analyze in order"""
        self.user_query = user_query
        self.think_and_print(f"🔍 Starting analysis for: '{user_query}'")
        self.think_and_print(f"🌐 Base URL: {start_url}")

        # Step 1: Crawl all internal pages
        self.crawl_site(start_url)

        if not self.all_urls:
            self.think_and_print("❌ No URLs discovered. Check connectivity or site structure.")
            return

        # Step 2: Rank by relevance
        ranked_urls = self.rank_urls()
        self.think_and_print(f"🎯 Ranked {len(ranked_urls)} URLs by relevance")

        # Step 3: Analyze in order — no early stopping
        self.think_and_print("🚀 Processing URLs from highest to lowest relevance...")
        for i, item in enumerate(ranked_urls, 1):
            url = item['url']
            self.think_and_print(f"📌 [{i}/{len(ranked_urls)}] Checking (Score: {item['score']}): {url}")
            self.crawl_and_analyze_page(url)
            time.sleep(1)  # Be respectful

        self.think_and_print(f"✅ Analysis complete. Visited {len(self.visited_urls)} pages.")

    def generate_final_report(self) -> str:
        """Generate final answer using all collected info"""
        if not self.relevant_info:
            return "❌ No relevant information found for your query."

        # Combine all info
        all_info = "\n\n---\n\n".join([
            f"Source: {item['url']}\n"
            f"Page Title: {item['title']}\n"
            f"Information:\n{item['information']}"
            for item in self.relevant_info
        ])

        prompt = f"""
        USER QUERY: {self.user_query}

        COMPREHENSIVE INFORMATION COLLECTED:
        {all_info}

        INSTRUCTIONS:
        - Synthesize a complete, well-structured answer.
        - Include all specific details: names, numbers, locations, products, emails, etc.
        - Remove duplicates.
        - Organize logically (e.g., by topic).
        - Be thorough and helpful.
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Final report generation failed: {e}")
            fallback = f"Found {len(self.relevant_info)} relevant pages:\n\n"
            for item in self.relevant_info:
                fallback += f"- {item['title']} ({item['url']}):\n  {item['information'][:300]}...\n\n"
            return fallback

    def cleanup(self):
        """Close browser"""
        if self.driver:
            self.driver.quit()


# ========================
#   MAIN EXECUTION
# ========================
def main():
    # ✅ YOUR CREDENTIALS
    PROJECT_ID = "wired-name-455213-q8"
    LOCATION = "us-west1"
    CREDENTIALS_PATH = "/Users/riddhi/Desktop/crawl/wired-name-455213-q8-41933c90f1bf.json"

    # Get user input
    START_URL = input("🌐 Enter the website URL to analyze: ").strip()
    USER_QUERY = input("❓ What information would you like to know? ").strip()

    # Validate URL
    if not START_URL.startswith(('http://', 'https://')):
        START_URL = 'https://' + START_URL

    # Initialize agent
    agent = SmartWebsiteAnalyzerAgent(
        project_id=PROJECT_ID,
        location=LOCATION,
        credentials_path=CREDENTIALS_PATH
    )

    try:
        agent.analyze_website(START_URL, USER_QUERY)

        print("\n" + "="*80)
        print("📊 FINAL ANALYSIS REPORT")
        print("="*80)
        print(f"Query: {USER_QUERY}")
        print(f"Website: {START_URL}")
        print("-"*80)
        print(agent.generate_final_report())
        print("\n" + "="*80)
        print(f"✅ Checked {len(agent.visited_urls)} pages. Found {len(agent.relevant_info)} relevant sources.")
        print("="*80)

    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        print(f"❌ Execution failed: {e}")
        print("💡 Check your internet, credentials, or website accessibility.")

    finally:
        agent.cleanup()


if __name__ == "__main__":
    main()