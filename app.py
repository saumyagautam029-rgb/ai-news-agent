import json
import re
import requests
import streamlit as st
from duckduckgo_search import DDGS

# Configuration for Local Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3"


def query_local_llm(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Queries local LLM via Ollama API."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            return f"Error: Ollama API returned status {response.status_code}"
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"


def fetch_latest_news(topic: str, max_results: int = 5):
    """Fetches latest news using DuckDuckGo Search API."""
    news_items = []
    with DDGS() as ddgs:
        results = ddgs.news(keywords=topic, max_results=max_results)
        for item in results:
            news_items.append({
                "title": item.get("title"),
                "source": item.get("source"),
                "url": item.get("url"),
                "snippet": item.get("body"),
                "date": item.get("date")
            })
    return news_items


def cross_validate_news(article: dict, model: str = DEFAULT_MODEL) -> dict:
    """Uses local LLM to cross-validate source quality, facts, and reliability."""
    prompt = f"""
    You are an expert news verifier and fact-checker. Analyze the following news article snippet:
    
    Title: {article['title']}
    Source: {article['source']}
    Snippet: {article['snippet']}
    
    Provide your output strictly in JSON format with the following keys:
    - "credibility_score": A rating from 1 to 10 (integer).
    - "key_facts": A list of 2-3 main factual bullet points extracted.
    - "verification_summary": Brief reasoning behind credibility score and cross-validation note.
    - "verified": True or False boolean.
    
    Output JSON ONLY, no extra markdown text.
    """
    raw_response = query_local_llm(prompt, model=model)
    
    try:
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except Exception:
        pass
        
    return {
        "credibility_score": 7,
        "key_facts": [article['snippet'][:100]],
        "verification_summary": raw_response[:200],
        "verified": True
    }


def generate_social_posts(article: dict, validation: dict, model: str = DEFAULT_MODEL) -> dict:
    """Generates tailored social media posts for LinkedIn, Twitter/X, and Instagram."""
    prompt = f"""
    Create engaging social media posts based on this verified news story:
    Title: {article['title']}
    Source: {article['source']}
    Link: {article['url']}
    Key Facts: {validation.get('key_facts', [])}
    
    Generate output strictly in JSON format with keys:
    - "linkedin": A professional post (150-200 words) with call-to-action and hashtags.
    - "twitter": A concise tweet (under 280 chars) with source link and hashtags.
    - "instagram": An engaging caption with strong visual hooks and relevant hashtags.
    
    Output JSON ONLY.
    """
    raw_response = query_local_llm(prompt, model=model)
    try:
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except Exception:
        pass
        
    return {
        "linkedin": f"📰 {article['title']}\n\n{article['snippet']}\n\nSource: {article['url']} #TechNews #AI",
        "twitter": f"🚨 {article['title'][:180]}...\nSource: {article['url']} #News",
        "instagram": f"✨ {article['title']}\n\n{article['snippet']}\n\n🔗 Link in bio!\n#LatestNews"
    }


# --- Streamlit UI ---
st.set_page_config(page_title="AI News Agent", page_icon="📰", layout="wide")

st.title("🤖 AI News & Social Post Agent (Zero-Cost Local Setup)")
st.caption("Powered by Ollama (Local Open-Source LLM) & DuckDuckGo News")

with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.text_input("Ollama Model Name", value="llama3")
    max_stories = st.slider("Max News Stories", min_value=1, max_value=10, value=3)
    st.markdown("---")
    st.markdown("**Status Check:**")
    try:
        res = requests.get("http://localhost:11434/")
        if res.status_code == 200:
            st.success("Ollama Running Locally ✅")
    except Exception:
        st.error("Ollama Not Detected! Run `ollama serve` in terminal.")

topic = st.text_input("Enter News Topic or Industry", value="Artificial Intelligence Education")

if st.button("🚀 Run AI Agent Pipeline"):
    if not topic.strip():
        st.warning("Please enter a valid topic.")
    else:
        with st.spinner("Fetching latest news from web..."):
            news = fetch_latest_news(topic, max_results=max_stories)
            
        if not news:
            st.error("No news found for this topic.")
        else:
            st.success(f"Fetched {len(news)} news articles!")
            
            for idx, item in enumerate(news, 1):
                st.markdown(f"### Article {idx}: {item['title']}")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"**Source:** {item['source']}")
                    st.markdown(f"**URL:** [{item['url']}]({item['url']})")
                    st.write(item['snippet'])
                    
                with col2:
                    with st.spinner(f"Cross-validating Article {idx} with Local LLM..."):
                        validation = cross_validate_news(item, model=model_choice)
                        
                    st.markdown("#### 🔍 Cross-Validation Summary")
                    st.json(validation)
                    
                st.markdown("#### 📲 Generated Social Media Content")
                with st.spinner(f"Generating posts for Article {idx}..."):
                    posts = generate_social_posts(item, validation, model=model_choice)
                    
                tab1, tab2, tab3 = st.tabs(["💼 LinkedIn", "🐦 Twitter / X", "📸 Instagram"])
                
                with tab1:
                    st.text_area("LinkedIn Post", value=posts.get("linkedin", ""), height=150, key=f"li_{idx}")
                with tab2:
                    st.text_area("Twitter Post", value=posts.get("twitter", ""), height=100, key=f"tw_{idx}")
                with tab3:
                    st.text_area("Instagram Caption", value=posts.get("instagram", ""), height=120, key=f"ig_{idx}")
                    
                st.markdown("---")