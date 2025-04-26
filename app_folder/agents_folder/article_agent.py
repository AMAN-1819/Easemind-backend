import warnings
warnings.filterwarnings("ignore")

from agno.agent import Agent, Message
import requests
import re
from app_folder.config import (
    GROQ_CLIENT, ARTICLE_MODEL, TAVILY_API_KEY,
    SERPER_API_KEY, RESOURCE_CACHE
)

class ArticleSuggestionAgent(Agent):
    def __init__(self, name="ArticleSuggestionAgent"):
        super().__init__(name=name)

    def is_quality_youtube_result(self, title, url):
        """Strict quality check for YouTube results"""
        if not title or not url:
            return False
        title = title.strip().lower()
        if len(title) < 15:  # Too short titles
            return False
        if title in ['youtube', 'video', 'watch', 'click here']:
            return False
        if not self.is_valid_youtube_url(url):
            return False
        return True

    def is_valid_youtube_url(self, url):
        youtube_regex = (
            r'(https?://)?(www\.)?'
            '(youtube|youtu|youtube-nocookie)\.(com|be)/'
            '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
        return re.match(youtube_regex, url) is not None

    def extract_tips(self, gita_response):
        lines = gita_response.strip().split('\n')
        tips = []
        current_tip = ""
        for line in lines:
            if line.strip() and line[0].isdigit() and "." in line[:5]:
                if current_tip:
                    tips.append(current_tip.strip())
                current_tip = line.strip()
            elif current_tip and line.strip():
                current_tip += " " + line.strip()
        if current_tip:
            tips.append(current_tip.strip())
        return tips[:3]

    def get_keywords(self, tip):
        cache_key = f"keywords_{tip[:50]}"
        if cache_key in RESOURCE_CACHE:
            return RESOURCE_CACHE[cache_key]
        
        prompt = f"""
        Extract 3-5 most important keywords from this Gita advice for web search:
        {tip}
        
        Return only comma-separated keywords (e.g., karma,duty,detachment)
        """
        try:
            response = GROQ_CLIENT.chat.completions.create(
                model=ARTICLE_MODEL,
                messages=[{"role": "system", "content": "Extract search keywords"},
                          {"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=50
            )
            keywords = response.choices[0].message.content.strip()
            RESOURCE_CACHE[cache_key] = keywords
            return keywords
        except Exception as e:
            print(f"Keyword extraction error: {e}")
            fallback = "bhagavad gita," + tip[:20].replace(' ', ',')
            RESOURCE_CACHE[cache_key] = fallback
            return fallback

    def search_resources(self, keywords):
        cache_key = f"resources_{keywords[:50]}"
        if cache_key in RESOURCE_CACHE:
            cached = RESOURCE_CACHE[cache_key]
            if cached["videos"] or cached["articles"]:
                return cached

        resources = {"videos": [], "articles": []}

        if TAVILY_API_KEY:
            try:
                response = requests.post(
                    "https://api.tavily.com/search",
                    json={
                        "query": f"{keywords} practical guide",
                        "include_domains": ["youtube.com", "edu", "org"],
                        "max_results": 5,
                        "api_key": TAVILY_API_KEY
                    },
                    timeout=10
                )
                if response.status_code == 200:
                    for result in response.json().get("results", []):
                        url = result.get("url", "")
                        title = result.get("title", "").strip()
                        if "youtube.com" in url:
                            if self.is_quality_youtube_result(title, url):
                                resources["videos"].append({"title": title, "url": url})
                        elif url.startswith(('http://', 'https://')):
                            if title and len(title) > 10:
                                resources["articles"].append({"title": title, "url": url})
            except Exception as e:
                print(f"Tavily error: {e}")

        if (not resources["videos"] and not resources["articles"]) and SERPER_API_KEY:
            try:
                response = requests.post(
                    "https://google.serper.dev/search",
                    json={"q": f"{keywords} site:youtube.com OR site:edu OR site:org"},
                    headers={
                        "X-API-KEY": SERPER_API_KEY,
                        "Content-Type": "application/json"
                    },
                    timeout=10
                )
                if response.status_code == 200:
                    for result in response.json().get("organic", []):
                        url = result.get("link", "")
                        title = result.get("title", "").strip()
                        if "youtube.com" in url:
                            if self.is_quality_youtube_result(title, url):
                                resources["videos"].append({"title": title, "url": url})
                        elif url.startswith(('http://', 'https://')):
                            if title and len(title) > 10:
                                resources["articles"].append({"title": title, "url": url})
            except Exception as e:
                print(f"Serper error: {e}")

        RESOURCE_CACHE[cache_key] = resources
        return resources

    def format_resources(self, all_videos, all_articles, seen_urls):
        lines = []
        
        # Always show articles section first
        lines.append("📘 Useful Articles:")
        if all_articles:
            for a in all_articles:
                if a["url"] not in seen_urls and a["title"]:
                    lines.append(f"{a['title']}")
                    lines.append(f"{a['url']}")
                    seen_urls.add(a["url"])
        else:
            lines.append("No relevant articles found")

        # Only show videos section if we have valid videos
        valid_videos = [v for v in all_videos 
                      if v["url"] not in seen_urls 
                      and self.is_quality_youtube_result(v["title"], v["url"])]
        
        if valid_videos:
            lines.append("\n📺 Helpful Videos:")
            for v in valid_videos[:3]:  # Limit to 3 best videos
                lines.append(f"{v['title']}")
                lines.append(f"{v['url']}")
                seen_urls.add(v["url"])
        elif all_videos:  # Had videos but none passed quality check
            lines.append("\n📺 Videos: No relevant videos found")

        return "\n".join(lines)

    def process_tip(self, tip, seen_urls):
        keywords = self.get_keywords(tip)
        resources = self.search_resources(keywords)
        return resources

    def run(self, message):
        if message.recipient != self.name:
            return None

        gita_advice = message.content
        tips = self.extract_tips(gita_advice)
        
        seen_urls = set()
        all_videos = []
        all_articles = []

        for tip in tips:
            resources = self.process_tip(tip, seen_urls)
            all_videos.extend(resources["videos"][:2])  # Get top 2 videos per tip
            all_articles.extend(resources["articles"][:2])  # Get top 2 articles per tip

        output = self.format_resources(all_videos, all_articles, seen_urls)

        return Message(role="assistant", content=output)