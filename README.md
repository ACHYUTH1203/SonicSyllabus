🎙️ SonicSyllabus: AI Audio Mentor (Kalam)
==========================================

**SonicSyllabus** is a high-fidelity, RAG-powered audio revision platform designed specifically for UPSC aspirants. By transforming complex constitutional documents into concise, conversational audio "bites," it allows students to master the Indian Constitution through a multi-language, eyes-free learning experience.

**Kalam**, the AI Mentor, bridges this gap by providing:

*   **Eyes-Free Revision**: Converts verified legal data into engaging 1-minute audio scripts.
    
*   **Zero Hallucination**: Employs a **Retrieval-Augmented Generation (RAG)** pipeline to ensure every word is grounded in the "Golden Data" of the Indian Constitution.
    
*   **Linguistic Inclusion**: Supports languages, including Hindi, Telugu, Kannada with precise legal terminology.
    

### 🖥️ Interface

### 🛠️ Technical Architecture

The core of SonicSyllabus is built using **LangGraph**, orchestrating a state-aware pipeline that handles everything from security to synthesis:

*   **State Management**: Uses a centralized GraphState to track query validity, retrieved context, and multi-language scripts.
    
*   **AI Security Guardrail**: A dedicated node at the entry point filters queries to ensure the mentor only discusses Constitutional Law, preventing API abuse and off-topic costs.
    
*   **Semantic Retrieval Engine**: Leverages text-embedding-3-small and a persistent **ChromaDB** vector store to retrieve high-relevance constitutional clauses.
    
*   **Contextual Generator**: A gpt-4o-mini node synthesizes retrieved data into a conversational script under 150 words, specifically formatted for natural Text-to-Speech pauses.
    
*   **Linguistic Architect**: A dynamic translation node that maintains legal accuracy while converting scripts into chosen regional languages.
    
*   **Neural Voice Synthesis**: Powered by OpenAI's tts-1 (Nova voice) for a realistic, human-like teaching performance.
    

### ✨ Key Features

*   **Premium Glassmorphism UI**: A modern, web-responsive https://www.google.com/search?q=interface featuring Gemini-inspired audio-reactive animations.
    
*   **Smart Audio Caching**: Implements semantic caching of .mp3 and .txt files to provide instant responses and zero-cost retrieval for repeated topics.
    
*   **Rate Limiting**: Integrated per-user session tracking (5 requests/day) to manage operational overhead and prevent API spam.
    
*   **One-Minute Bites**: Strict word-count constraints ensure the LLM and TTS respond in under 20 seconds.
    

### ⚙️ Setup & Execution

#### **Prerequisites**

*   Python 3.10+
    
*   Poetry (Recommended)
    
*   OpenAI API Key
    

#### **1\. Environment Setup**

Create a .env file in the root directory:

Code snippet

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   OPENAI_API_KEY=your_actual_key_here   `

#### **2\. One-Command Execution**

The following command installs all dependencies, ingests the Constitutional data into your local vector database, and starts the web server:

Bash

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   poetry install && poetry run python ingest.py && poetry run python app.py   `

> **Note**: Once the server starts, open your browser and navigate to http://localhost:8000 to begin your revision session.