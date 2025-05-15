# app_folder/agents_folder/gita_agent.py
# import warnings
# warnings.filterwarnings("ignore")
from fastembed import TextEmbedding  # New (recommended)

from agno.agent import Agent, Message
from app_folder.config import (
    CLIENT, EMBED_MODEL, GROQ_CLIENT, GITA_MODEL,
    COLLECTION_NAME, GITA_CACHE
)

class GitaAIAgent(Agent):
    def __init__(self, name="GitaAIAgent"):
        super().__init__(name=name)

    def dig_into_gita(self, question, k=3):
        cache_key = f"{question}_{k}"
        if cache_key in GITA_CACHE:
            return GITA_CACHE[cache_key]
        question_embedding = list(EMBED_MODEL.embed([question]))[0]
        results = CLIENT.query_points(
            collection_name=COLLECTION_NAME,
            query=question_embedding,
            limit=k,
            with_payload=True
        ).points
        GITA_CACHE[cache_key] = results
        if len(GITA_CACHE) > 100:
            GITA_CACHE.pop(next(iter(GITA_CACHE)))
        return results

    def cook_up_an_answer(self, question, gita_bits):
        def smart_truncate(text, max_length=40):
            if len(text) <= max_length:
                return text
            return text[:max_length].rsplit(' ', 1)[0] + '...'

        gita_text = "\n".join([smart_truncate(bit.payload["context"]) for bit in gita_bits]) if gita_bits else "No answer in Shri bhagawad Gita."

        prompt = f"""
        You're helping a stressed student with Gita wisdom.Speak like a calm, caring friend — thoughtful, simple, and natural.Keep it short(upto 2 lines):
        1. Understand the question: {question}
        2. Provide exactly 3-4 tips
        3. Format each tip as:
           [#].[4-5 line casual explanation]
           Chapter [X], Verse [Y]
        4. NO BOLD TEXT ALLOWED - use plain text only
        5. Ensure both tips are complete with full explanations(variety ,sophistication,fluidity) and don't be repetative
        Gita Context: {gita_text}
        """

        try:
            response = GROQ_CLIENT.chat.completions.create(
                model=GITA_MODEL,
                messages=[
                    {"role": "system", "content": "DO NOT USE BOLD TEXT OR MARKDOWN. Output must be completely plain text only with no formatting, no asterisks, no stars, no markdown symbols. Provide exactly 3-4 complete Gita-based tips."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.4,
                top_p=0.85
            )

            content = response.choices[0].message.content.strip()
            content = content.replace("**", "").replace("*", "")
            return content
        except Exception as e:
            return f"Sorry, I couldn't retrieve Gita wisdom at the moment. Error: {str(e)}"

    def run(self, message):
        question = message.content
        gita_bits = self.dig_into_gita(question, k=3)

        if not gita_bits:
            answer = "It doesn't have direct answer in Shri Bhagawad Gita"
        else:
            answer = self.cook_up_an_answer(question, gita_bits)

        check_prompt = f"Advice: '{answer}'. Verify format: '[#].explanation' ' then 'Chapter [X], Verse [Y]'. Keep it chill, clear, no bold, no stars,The output should be in same language as of user, refine if needed."
        check = GROQ_CLIENT.chat.completions.create(
            model=GITA_MODEL,
            messages=[{"role": "system", "content": "Check format, refine if needed, no bold, no stars."}, {"role": "user", "content": check_prompt}],
            max_tokens=600,
            temperature=0.6
        )
        refined = check.choices[0].message.content.strip()
        if "refine" in refined.lower() or "tweak" in refined.lower():
            answer = refined

        return Message(role="assistant", content=answer, recipient="ArticleSuggestionAgent")