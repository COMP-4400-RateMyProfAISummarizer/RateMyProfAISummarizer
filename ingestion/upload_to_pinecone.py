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
        print(f"🚀 Direct Access to {prof_name}: {prof_url}")
        
        try:
            response = requests.get(prof_url, headers=self.headers, timeout=10)
            tree = html.fromstring(response.content)
            
            # Scrape Stats
            avg_rating = tree.xpath('//div[contains(@class, "RatingValue__Numerator")]/text()')[0]
            feedback = tree.xpath('//div[contains(@class, "FeedbackItem__FeedbackNumber")]/text()')
            
            wta = feedback[0] if len(feedback) > 0 else "N/A"
            avg_diff = feedback[1] if len(feedback) > 1 else "N/A"
            
            # Scrape Comments
            comments = tree.xpath('//div[contains(@class, "Comments__StyledComments")]/text()')
            
            return {
                "name": prof_name,
                "avgRating": avg_rating,
                "wouldTakeAgain": wta,
                "avgDifficulty": avg_diff,
                "reviews": list(set(comments))
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
    
    with open('data/cs_professors.json', 'r') as f:
        target_profs = json.load(f)
    
    all_texts = []
    all_metadatas = []
    
    # --- COUNTERS ---
    total_reviews_processed = 0
    total_profs_scanned = 0

    for item in target_profs:
        data = scraper.get_professor_by_id(item['id'], item['name'])
        total_profs_scanned += 1
        
        if data and data['reviews']:
            review_count = len(data['reviews'])
            total_reviews_processed += review_count
            print(f"✅ Processing {data['name']} ({item['dept']}) | {review_count} reviews found.")
            
            for review_text in data['reviews']:
                if len(review_text.strip()) < 20:
                    continue
                
                chunks = text_splitter.split_text(review_text)
                
                for chunk in chunks:
                    all_texts.append(chunk)
                    all_metadatas.append({
                        "prof_name": data['name'],
                        "dept": item['dept'],
                        "prof_id": item['id'],
                        "avg_rating": data['avgRating'],
                        "would_take_again": data['wouldTakeAgain'],
                        "avg_difficulty": data['avgDifficulty']
                    })
        
        time.sleep(1.5)

    # Final Statistics Output
    print("\n--- 📊 INGESTION SUMMARY ---")
    print(f"Total Professors Scanned: {total_profs_scanned}")
    print(f"Total Raw Reviews Processed: {total_reviews_processed}")
    print(f"Total Semantic Chunks Created: {len(all_texts)}")
    print("----------------------------\n")

    if all_texts:
        print(f"📤 Upserting to Pinecone index: {os.getenv('PINECONE_INDEX_NAME')}...")
        vector_store = PineconeVectorStore.from_texts(
            texts=all_texts,
            embedding=embeddings,
            metadatas=all_metadatas,
            index_name=os.getenv("PINECONE_INDEX_NAME")
        )
        print("🎉 SUCCESS: Vectors and Metadata are live.")

if __name__ == "__main__":
    run_verified_ingestion()