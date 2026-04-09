# SonicSyllabus: AI Audio Mentor (Kalam)

**SonicSyllabus** is a high-fidelity, RAG-powered audio revision platform designed specifically for UPSC aspirants. By transforming complex constitutional documents into concise, conversational audio "bites," it allows students to master the Indian Constitution through a multi-language, eyes-free learning experience.

**Kalam**, the AI Mentor, bridges this gap by providing:

- **Eyes-Free Revision**: Converts verified legal data into engaging 1-minute audio scripts.
- **Zero Hallucination**: Employs a **Retrieval-Augmented Generation (RAG)** pipeline to ensure every word is grounded in the "Golden Data" of the Indian Constitution.
- **Linguistic Inclusion**: Supports 13 languages — Hindi, Telugu, Tamil, Kannada, Malayalam, Marathi, Bengali, Gujarati, Punjabi, Odia, Assamese, and Urdu — with precise legal terminology.

---

## Technical Architecture

The core of SonicSyllabus is built using **LangGraph**, orchestrating a state-aware pipeline that handles everything from security to synthesis:

- **State Management**: Uses a centralized `GraphState` to track query validity, retrieved context, and multi-language scripts.
- **AI Security Guardrail**: A dedicated entry-point node filters queries to ensure the mentor only discusses Constitutional Law, preventing API abuse and off-topic costs.
- **Semantic Retrieval Engine**: Leverages `text-embedding-3-small` and a persistent **ChromaDB** vector store to retrieve high-relevance constitutional clauses.
- **Document Grader**: Evaluates retrieved chunks for relevance before passing them to the generator, reducing noise in the context window.
- **Contextual Generator**: A `gpt-4o-mini` node synthesizes retrieved data into a conversational script under 150 words, formatted for natural Text-to-Speech pauses.
- **Linguistic Architect**: A dynamic translation node that maintains legal accuracy while converting scripts into the chosen regional language.
- **Neural Voice Synthesis**: Powered by OpenAI's `tts-1-hd` (Nova voice) for high-definition, human-like teaching audio.

### Pipeline Flow

```
guardrail → retrieve → grade_documents → generate_english → [translate_dynamic] → END
```

---

## Key Features

- **Premium Glassmorphism UI**: A modern, web-responsive interface featuring audio-reactive animations.
- **Smart Audio + Text Caching**: Caches both `.mp3` and `.txt` files per topic/language pair, returning the script text alongside audio for instant, zero-cost retrieval on repeated queries.
- **Rate Limiting**: Per-user session tracking (5 requests/day) to manage operational overhead and prevent API spam.
- **One-Minute Bites**: Strict word-count constraints ensure the LLM and TTS respond in under 20 seconds.

---

## Setup & Execution

### Prerequisites

- Python 3.11+
- Poetry
- OpenAI API Key

### 1. Environment Setup

Create a `.env` file in the root directory:

```
OPENAI_API_KEY=your_actual_key_here
```

### 2. One-Command Execution

The following command installs dependencies, ingests the Constitutional data into the vector database, and starts the server:

```bash
poetry install && poetry run python ingest.py && poetry run python app.py
```

> **Note:** Once the server starts, open your browser and go to [http://localhost:8000](http://localhost:8000) to begin your revision session.
