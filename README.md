# KPMG GenAI Developer Home Assignment 

This repository contains the completed assignment for the KPMG GenAI Developer Assessment, which includes two parts:

1. **Phase 1**: Form extraction from National Insurance Institute (ביטוח לאומי) forms using OCR and Azure OpenAI
2. **Phase 2**: Microservice-based chatbot for medical services information for Israeli health funds

## Setup

### Prerequisites
- Python 3.9 or higher
- Azure OpenAI API access
- Azure Document Intelligence (Form Recognizer) access

### Installation

1. Clone this repository
   ```
   git clone https://github.com/Mayarozin0/Home-Assiggnment-Maya-Rozin.git
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

## Environment Setup

This project requires separate environment configurations for each phase:

### Phase 1 Environment
- Create a `.env` file in the `phase1/` directory based on the provided template:
  ```
  AZURE_FORM_RECOGNIZER_ENDPOINT=
  AZURE_FORM_RECOGNIZER_KEY=
  AZURE_OPENAI_API_KEY=
  AZURE_OPENAI_ENDPOINT=
  ```

### Phase 2 Environment
- Create a `.env` file in the `phase2/medical-services-chatbot/` directory based on the provided template:
  ```
  AZURE_OPENAI_API_KEY=
  AZURE_OPENAI_ENDPOINT=
  AZURE_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
  AZURE_GPT4_DEPLOYMENT=gpt-4o
  AZURE_GPT4_MINI_DEPLOYMENT=gpt-4o-mini
  ```

Fill each `.env` file with the appropriate Azure credentials provided to you. These environment files should not be committed to the repository (they are included in `.gitignore`).

## Running Phase 1: Form Extraction

Phase 1 implements a Streamlit application that extracts information from National Insurance Institute forms.

To run the Phase 1 application:
```
streamlit run phase1/phase1_main.py
```

This will start a Streamlit application on http://localhost:8501 where you can:
1. Upload a PDF/JPG form
2. View the extracted data in JSON format
3. Download the extracted data

## Running Phase 2: Medical Services Chatbot

Phase 2 implements a microservice-based chatbot with separate backend and frontend components.

### Starting the Backend
```
python phase2/medical-services-chatbot/backend/main.py
```
This will start the FastAPI service on http://localhost:8000

### Starting the Frontend
In a new terminal:
```
streamlit run phase2/medical-services-chatbot/frontend/app.py
```
This will start the Streamlit interface on http://localhost:8501

The chatbot features:
1. Two phases: information collection and Q&A
2. Multi-language support (Hebrew and English)
3. HMO-specific information retrieval

## Project Structure
- `phase1/` - Form extraction implementation
- `phase2/medical-services-chatbot/` - Chatbot implementation
  - `backend/` - FastAPI microservice
  - `frontend/` - Streamlit UI
  - `data/` - Knowledge base and vector store

## Notes
- The vector store is pre-built and included in the repository
- If you need to rebuild the vector store, run:
  ```
  cd phase2/medical-services-chatbot
  python backend/knowledge_base.py
  python backend/embed_knowledge_base.py
  ```

## Note About Phase 1

While the extraction works for most data points, I'm aware that it has some limitations. It occasionally drops the last digit of ID numbers, sometimes misidentifies signatures, and can get confused with different address components (apartment numbers, entrances, etc.).

This is my first time working with OCR and Document Intelligence, and given the time constraints of this assignment, I decided to stop at this implementation level. In a real-world project, I would have explored custom form models in Azure Document Intelligence, which would likely yield more accurate results for these specific forms. This approach would require more setup and training, but would probably deliver better extraction precision.
