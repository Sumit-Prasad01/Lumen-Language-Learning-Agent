import os
import re
import json
import random

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

translation_model = ChatOllama(
    model="llama3.2:3b",
    temperature=0.7
)

@tool
def get_n_random_words(language: str,
                       n: int, ) -> list:
    """
    Selects a specified number of random words from a language-specific word list.

    The function reads a JSON file containing words for the specified language from
    a predefined directory. It then selects `n` random words from the file and
    returns them in a list.

    :param language: A string representing the language for which to fetch the word list.
    :param n: An integer specifying the number of random words to retrieve.
    :return: A list containing `n` randomly selected words.
    """
    path = os.path.join("data", f"{language}", "word-list-cleaned.json")

    with open(path) as f:
        word_list = json.load(f)

    random_word_dict = {k: word_list[k] for k in random.sample(list(word_list.keys()), n)}
    random_words = [item["word"] for item in random_word_dict.values()]

    return random_words

@tool
def get_n_random_words_by_difficulty_level(language: str,
                                           difficulty_level: str,
                                           n: int
                                           ) -> list:
    """
    Retrieves a specified number of random words filtered by a given difficulty level
    from a word list corresponding to a specific language. The function reads the
    word list from a JSON file located in the directory `data/{language}/word-list-cleaned.json`.

    :param language: The language of the word list to be used.
    :type language: str
    :param difficulty_level: The difficulty level to filter words by. Possible values
        depend on the data structure in the JSON file. The only valid values are "beginner",
        "intermediate" and "advanced".
    :type difficulty_level: str
    :param n: The number of random words to retrieve.
    :type n: int
    :return: A list containing `n` random words filtered by the specified difficulty level.
    :rtype: list
    """
    path = os.path.join("data", f"{language}", "word-list-cleaned.json")

    with open(path) as f:
        word_list = json.load(f)

    words_filtered_by_difficulty = {k: v for k, v in word_list.items() if v.get("word_difficulty") == difficulty_level}

    random_word_dict = {k: words_filtered_by_difficulty[k] for k in
                        random.sample(list(words_filtered_by_difficulty.keys()), n)}
    random_words = [item["word"] for item in random_word_dict.values()]

    return random_words

@tool
def translate_words(random_words: list[str],
                    source_language: str,
                    target_language: str) -> dict:
    """
        Translates a list of words from a source language to a target language using
        a language model. The method ensures output is in the expected JSON format,
        containing translations corresponding to the provided input words.

        :param random_words: A list of words to be translated.
        :param source_language: The language of the input words.
        :param target_language: The language to translate the words into.
        :return: A dictionary containing the translations with the structure:
                {"translations": [{"source": "<original>", "target": "<translated>"}, ...]}.
    """

    prompt = (
        f"You are a precise translation engine.\n"
        f"Translate each of the following {len(random_words)} words from {source_language} to {target_language}.\n"
        f"Return ONLY valid JSON with this exact structure:\n"
        f'{{"translations": [{{"source": "<original>", "target": "<translated>"}}, ...]}}\n'
        f"STRICT RULES:\n"
        f"- Use double quotes only\n"
        f"- No markdown\n"
        f"- No explanations\n"
        f"- No trailing commas\n"
        f"Words: {json.dumps(random_words, ensure_ascii=False)}"
    )

    response = translation_model.invoke([HumanMessage(content=prompt)])
    text = getattr(response, "content", str(response)).strip()

    def extract_and_fix_json(raw: str) -> dict:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {}

        json_text = match.group(0).strip()

        # Fix common LLM JSON mistakes
        json_text = json_text.replace("'", '"')  # single quotes -> double quotes
        json_text = re.sub(r",\s*}", "}", json_text)  # trailing commas
        json_text = re.sub(r",\s*]", "]", json_text)

        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            return {}

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = extract_and_fix_json(text)

    translations_list = parsed.get("translations", [])

    model_map = {
        item.get("source", ""): item.get("target", "")
        for item in translations_list
        if isinstance(item, dict)
    }

    ordered_translations = [
        {"source": w, "target": model_map.get(w, model_map.get(w.capitalize(), w))}
        for w in random_words
    ]

    return {"translations": ordered_translations}