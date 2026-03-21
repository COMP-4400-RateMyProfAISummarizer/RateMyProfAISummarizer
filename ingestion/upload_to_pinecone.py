import requests
import os
import time
import json
from lxml import html
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

class UWindsorScraper:
    def __init__(self):
        self.headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

    def get_professor_by_id(self, prof_id, prof_name):
        prof_url = f"https://www.ratemyprofessors.com/professor/{prof_id}"
        print(f"🚀 Accessing {prof_name}: {prof_url}")
        
        try:
            response = requests.get(prof_url, headers=self.headers, timeout=10)
            tree = html.fromstring(response.content)
            
            # Scrape Stats - Using safe indexing to avoid crashes
            rating_elements = tree.xpath('//div[contains(@class, "RatingValue__Numerator")]/text()')
            avg_rating = rating_elements[0] if rating_elements else "N/A"
            
            feedback = tree.xpath('//div[contains(@class, "FeedbackItem__FeedbackNumber")]/text()')
            wta = feedback[0] if len(feedback) > 0 else "N/A"
            avg_diff = feedback[1] if len(feedback) > 1 else "N/A"
            
            # Scrape Comments
            comments = tree.xpath('//div[contains(@class, "Comments__StyledComments")]/text()')
            unique_comments = list(set(comments))
            
            return {
                "name": prof_name,
                "avgRating": avg_rating,
                "wouldTakeAgain": wta,
                "avgDifficulty": avg_diff,
                "reviews": unique_comments
            }
        except Exception as e:
            print(f"❌ Error fetching ID {prof_id}: {e}")
            return None

def run_verified_ingestion():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    
    scraper = UWindsorScraper()
    
    # Ensure data directory exists
    if not os.path.exists('data/professors.json'):
        print("❌ Error: data/professors.json not found!")
        return

    with open('data/professors.json', 'r') as f:
        target_profs = json.load(f)
    
    all_texts = []
    all_metadatas = []
    
    total_reviews_processed = 0
    total_profs_scanned = 0

    for item in target_profs:
        data = scraper.get_professor_by_id(item['id'], item['name'])
        total_profs_scanned += 1
        
        if not data:
            continue

        # --- FALLBACK LOGIC ---
        # If no reviews exist, create a placeholder so the metadata is still searchable
        if not data['reviews']:
            print(f"⚠️ {data['name']} has 0 reviews. Injecting placeholder metadata.")
            reviews_to_process = ["No detailed student reviews are currently available for this professor."]
        else:
            reviews_to_process = data['reviews']
            print(f"✅ Processing {data['name']} | {len(reviews_to_process)} reviews found.")

        review_count = 0
        for review_text in reviews_to_process:
            clean_text = review_text.strip()
            
            # Skip short junk reviews, but ALWAYS allow the placeholder text
            if len(clean_text) < 20 and "No detailed student reviews" not in clean_text:
                continue
            
            review_count += 1
            chunks = text_splitter.split_text(clean_text)
            
            for chunk in chunks:
                all_texts.append(chunk)
                all_metadatas.append({
                    "prof_name": data['name'],
                    "dept": item.get('dept', 'Unknown'),
                    "prof_id": item['id'],
                    "avg_rating": data['avgRating'],
                    "would_take_again": data['wouldTakeAgain'],
                    "avg_difficulty": data['avgDifficulty']
                })
        
        total_reviews_processed += review_count
        time.sleep(1.2) # Polite delay

    # Statistics
    print("\n--- 📊 INGESTION SUMMARY ---")
    print(f"Total Professors Scanned: {total_profs_scanned}")
    print(f"Total Profiles with Entries: {len(set([m['prof_name'] for m in all_metadatas]))}")
    print(f"Total Semantic Chunks Created: {len(all_texts)}")
    print("----------------------------\n")

    if all_texts:
        print(f"📤 Upserting to Pinecone index: {os.getenv('PINECONE_INDEX_NAME')}...")
        vector_store = PineconeVectorStore.from_texts(
            texts=all_texts,
            embedding=embeddings,
            metadatas=all_metadatas,
            index_name=os.getenv("PINECONE_INDEX_NAME"),
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )
        print("🎉 SUCCESS: Vectors and Metadata are live.")

if __name__ == "__main__":
    run_verified_ingestion()