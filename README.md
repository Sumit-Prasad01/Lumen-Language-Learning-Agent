# Lumen – Language Learning Agent 🚀

Lumen is an **LLM-powered Language Learning Agent** built to automate vocabulary learning workflows.
It can generate vocabulary lists, translate words, and automatically create **Anki decks & flashcards** using **Model Context Protocol (MCP)**.

This project is designed as an end-to-end pipeline, starting from raw multilingual word lists → cleaning & preprocessing → difficulty classification → interactive ReAct agent → Anki deck automation.

---

## ✨ Features

- Generate random vocabulary words in supported languages  
- Generate difficulty-aware vocabulary lists (**beginner / intermediate / advanced**)  
- Translate generated vocabulary into a target language  
- Create and manage Anki decks automatically  
- Create flashcards inside Anki using MCP integration (AnkiConnect backend)  
- Built using **LangGraph ReAct Agent workflow**  
- Supports Groq + Ollama multi-model architecture  
- Includes complete NLP preprocessing pipeline for cleaning word datasets  

---

## 🧠 Tech Stack

### Agent / LLM Framework
- **LangGraph**
- **LangChain**
- ReAct Agent Pattern

### LLM Providers
- **Groq API** (`llama-3.3-70b-versatile`) → reasoning + tool calling  
- **Ollama** (`llama3.2:3b`) → translations (local model)  

### NLP + Data Processing
- spaCy
- wordfreq
- Zipf’s Law based filtering
- Lemmatization pipeline
- JSON based cleaned word lists

### Anki Integration
- **AnkiConnect**
- **Model Context Protocol (MCP)**
- **clanki** MCP server

### Backend / API
- FastAPI (for sending prompts to the agent)

---

## 📂 Project Structure

```
Lumen-Language-Learning-Agent/
│
├── agent/                      # LangGraph agent logic + tools
│   ├── __init__.py
│   └── tools.py
│
├── clanki/                      # MCP server integration for Anki automation
│
├── config/                      # Configuration files
│
├── data/                        # Cleaned JSON word datasets (language-wise)
│
├── logs/                        # Logging output
│
├── notebooks/                   # Experiments / pipeline notebooks
│
├── pipeline/                    # NLP preprocessing pipeline modules
│
├── raw-word-list/               # Raw multilingual vocabulary word lists
│
├── src/                         # Scripts for data ingestion and processing
│   ├── data_ingestion.py
│   ├── data_processor.py
│   └── download_spacy_models.py
│
├── utils/                       # Helper utilities (logging, configs, etc.)
│
├── assistant-groq.py            # Agent execution using Groq LLM
├── assistant-ollama.py          # Agent execution using Ollama LLM
│
├── requirements.txt
├── pyproject.toml
├── setup.py
└── README.md
```

---

## 🌍 Supported Languages

Currently supported languages:

- **English**
- **German**
- **Spanish**

Datasets for these languages are stored inside:

```
data/<language>/word-list-cleaned.json
```

---

## 🛠 Custom Tools

The agent uses 3 core custom tools:

### 1️⃣ `get_n_random_words`
Fetches **N random words** from the cleaned dataset of a given language.

**Example:**
> Get 10 words in German.

---

### 2️⃣ `get_n_random_words_by_difficulty_level`
Fetches **N random words filtered by difficulty level**.

Difficulty levels:
- beginner
- intermediate
- advanced

**Example:**
> Get 15 beginner words in Spanish.

---

### 3️⃣ `translate_words`
Translates a list of words from a **source language** to a **target language** using an LLM.

**Example:**
> Translate 10 Spanish words to English.

---

## 📌 Prerequisites

Before running the agent, make sure you have:

### ✅ Python
- Python **3.10+** recommended (works on 3.12 too)

### ✅ Node.js
- Node.js **16+** required (for MCP clanki integration)

### ✅ Anki
- Install Anki Desktop
- Install **AnkiConnect** add-on  
  Add-on code:

```
2055492159
```

Restart Anki after installation.

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Sumit-Prasad01/Lumen-Language-Learning-Agent.git
cd Lumen-Language-Learning-Agent
```

### 2️⃣ Create virtual environment
```bash
python -m venv .lumen-env
source .lumen-env/bin/activate   # Linux/Mac
.lumen-env\Scripts\activate    # Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Setup

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🧪 Running the Agent

### ▶ Run with Groq (Reasoning + Tool Calling)
```bash
python assistant-groq.py
```

### ▶ Run with Ollama (Local Translation Model)
Make sure Ollama is installed and model is pulled:

```bash
ollama pull llama3.2:3b
```

Then run:

```bash
python assistant-ollama.py
```

---

## 🧩 Anki Deck Automation (MCP + clanki)

This project uses **Model Context Protocol (MCP)** with the **clanki** MCP server to interact with Anki.

Workflow:
1. Agent generates vocabulary words  
2. Agent optionally translates them  
3. Agent creates an Anki deck (`create-deck`)  
4. Agent creates flashcards (`create-card`)  

⚠️ Important:
- Anki must be open and running.
- AnkiConnect must be installed.
- AnkiConnect runs at:

```
http://127.0.0.1:8765
```

---

## 🧠 NLP Data Cleaning Pipeline

The pipeline converts raw word lists into structured cleaned JSON datasets.

Key steps:

- Inspect and debug raw vocabulary data
- Remove noise and invalid tokens
- Lemmatize words using spaCy transformer models
- Filter rare/uncommon words using Zipf’s Law
- Frequency analysis using `wordfreq`
- Build a full NLP pipeline
- Convert cleaned data into JSON
- Validate results with Spanish dataset
- Compare raw vs cleaned data

---

## 🧪 Example Prompts

### Random words
```
Get 10 random words in English
```

### Difficulty-based words
```
Get 20 beginner words in Spanish
```

### Translation
```
Get 15 intermediate words in German and translate them to English
```

### Create Anki Deck
```
Get 20 easy words in Spanish, translate them to English, and create a new Anki deck called Spanish::Easy
```

---

## 🌐 FastAPI Integration

A FastAPI app is included to send user prompts to the agent programmatically.

This can be extended for:
- frontend integration
- chatbot UI
- API-based Anki automation
- language learning assistant apps

---

## 📌 Roadmap / Future Improvements

- Add more supported languages
- Add part-of-speech based filtering (noun/verb/adjective)
- Add spaced repetition scheduling support
- Add vocabulary quizzes & tests
- Add UI dashboard for learners
- Add export to CSV / PDF
- Improve tool calling reliability across providers

---

## 🧾 Credits

This project integrates:
- LangGraph + LangChain ecosystem
- Groq LLM API
- Ollama local LLM inference
- spaCy + wordfreq for NLP preprocessing
- MCP + clanki for Anki automation

---


