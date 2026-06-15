# 🤖 Production RAG Chatbot with MLflow & SQLite

A **multi-user, enterprise-grade Retrieval-Augmented Generation (RAG) chatbot** built with Streamlit, Azure OpenAI (GPT-4o), ChromaDB, SQLite, and MLflow. Upload your documents — PDFs, Word files, text files, or images — and chat with an AI that answers questions grounded in your private knowledge base, with full production observability and human feedback tracking.

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| 📄 **Multi-Format Document Ingestion** | Upload PDFs, `.docx`, `.txt`, and images (`.png`, `.jpg`) with automatic OCR via Tesseract |
| 🔍 **RAG-Powered Q&A** | Answers are grounded in your uploaded documents using vector similarity search |
| 🤖 **Intelligent Fallback** | If the answer isn't in your docs, the system falls back to general AI knowledge and tells you |
| 👥 **Multi-User Authentication** | Register/login system with per-user session isolation and chat history |
| 💬 **Persistent Chat History** | All conversations are stored in SQLite and accessible from the sidebar |
| 👍👎 **RLHF Feedback Loop** | Thumbs up/down feedback that triggers creative retries and logs to MLflow |
| 📊 **Full MLflow Observability** | Every interaction is traced — latency, token usage, retrieval success, and human evaluations |
| 📦 **Dataset Export** | One-click compilation of interaction data into evaluation-ready CSV/JSON datasets |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                      │
│              (Chat UI · Sidebar · Auth Pages)               │
├─────────────────────────────────────────────────────────────┤
│                       Core Logic                            │
│   ┌──────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│   │ Chat     │  │ RAG Pipeline │  │ Feedback (RLHF)      │ │
│   │ Engine   │  │ + File Loader│  │ + Dataset Export      │ │
│   └────┬─────┘  └──────┬───────┘  └──────────┬───────────┘ │
├────────┼────────────────┼────────────────────┼──────────────┤
│        ▼                ▼                    ▼              │
│  ┌──────────┐    ┌───────────┐    ┌──────────────────────┐ │
│  │Azure     │    │ ChromaDB  │    │ MLflow Tracking      │ │
│  │OpenAI    │    │ (Vectors) │    │ Server               │ │
│  │GPT-4o    │    └───────────┘    └──────────────────────┘ │
│  └──────────┘                                              │
│                    ┌───────────┐                            │
│                    │  SQLite   │                            │
│                    │ (Users,   │                            │
│                    │  History, │                            │
│                    │  Feedback)│                            │
│                    └───────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Role |
| :--- | :--- | :--- |
| **Frontend** | [Streamlit](https://streamlit.io/) | Interactive chat interface and sidebar controls |
| **LLM** | [Azure OpenAI GPT-4o](https://azure.microsoft.com/en-us/products/ai-services/openai-service) | Dual-mode inference — precise (temp 0.3) and creative (temp 0.9) |
| **Embeddings** | Azure OpenAI `text-embedding-ada-002` | Converts document chunks into vector representations |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) | Persistent vector database for semantic document search |
| **Relational DB** | [SQLite](https://www.sqlite.org/) | User accounts, chat history, and feedback storage |
| **Orchestration** | [LangChain](https://www.langchain.com/) | Document loading, text splitting, retriever chain |
| **Observability** | [MLflow](https://mlflow.org/) | Experiment tracking, trace logging, dataset management |
| **OCR** | [Tesseract](https://github.com/tesseract-ocr/tesseract) + Pillow | Optical character recognition for image-based documents |

---

## 📁 Project Structure

```
Chatgpt-rag-app-mlflow-sqlite/
│
├── app.py                    # Main entry point — chat loop, MLflow logging, session management
├── config.py                 # Central configuration (Azure credentials, DB paths)
├── db.py                     # Database gateway (ChromaDB + SQLite initialization)
├── extract_dataset.py        # MLflow dataset extraction and CSV export utility
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (Azure API keys)
├── .gitignore                # Git ignore rules
│
├── auth/                     # Authentication module
│   ├── login.py              #   Login form and credential validation
│   └── register.py           #   User registration with duplicate checking
│
├── chat/                     # Chat interface module
│   ├── chat_engine.py        #   Dual LLM configuration (normal + creative)
│   ├── chat_ui.py            #   Sidebar rendering, file upload, history navigation
│   └── history.py            #   Chat history retrieval from SQLite
│
├── rag/                      # RAG pipeline module
│   ├── rag_pipeline.py       #   Document chunking, embedding, and vector store indexing
│   └── file_loader.py        #   Multi-format file parser (PDF, DOCX, TXT, images/OCR)
│
├── feedback/                 # Human feedback module
│   └── rlhf.py               #   Thumbs up/down handling, MLflow trace assessments
│
├── utils/                    # Utility module
│   └── session.py            #   Streamlit session state initialization
│
├── datasets/                 # Generated evaluation datasets (CSV)
├── ScreenShots/              # Application screenshots
├── chroma_db7/               # ChromaDB persistent storage
├── mlartifacts/              # MLflow artifact storage
├── BEGINNERS_GUIDE.md        # Detailed beginner-friendly technical guide
└── Users.db                  # SQLite database file
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+**
- **Azure OpenAI** resource with deployed `gpt-4o` and `text-embedding-ada-002` models
- **Tesseract OCR** installed on your system ([installation guide](https://github.com/tesseract-ocr/tesseract#installing-tesseract))

### 1. Clone the Repository

```bash
git clone https://github.com/Likithkumarr/Chatgpt-Rag-App-mlflow.git
cd Chatgpt-Rag-App-mlflow
```

### 2. Create a Virtual Environment

```bash
python -m venv myenv
myenv\Scripts\activate        # Windows
# source myenv/bin/activate   # macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root with your Azure OpenAI credentials:

```env
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=Version
AZURE_OPENAI_CHAT_DEPLOYMENT=model
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=embedding-model
```

### 5. Start the MLflow Tracking Server

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

### 6. Launch the Application

In a separate terminal:

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**.

---

## 📖 Usage Guide

### Registration & Login
1. Open the app and navigate to the **Register** tab to create an account.
2. Switch to the **Login** tab to sign in with your credentials.

### Uploading Documents
1. Use the **sidebar file uploader** to upload PDFs, Word documents, text files, or images.
2. The system will automatically chunk, embed, and index the content into ChromaDB.
3. A progress bar tracks the vectorization process.

### Chatting with Your Documents
- Type your question in the chat input. The system will:
  - **Search** your uploaded documents for relevant context.
  - **Answer using document context** if a match is found (indicated by 📄).
  - **Fall back to general AI** if no relevant context exists (indicated by 🤖).

### Providing Feedback (RLHF)
- **👍 Thumbs Up** — Saves the interaction as a positive training signal.
- **👎 Thumbs Down** — Discards the bad response and automatically retries with a creative LLM.

### Exporting Evaluation Datasets
- Click **"🔄 Compile Latest Evaluation Dataset"** in the sidebar to export your interactions (with feedback scores) as a CSV file into the `datasets/` directory.

---

## 📊 MLflow Monitoring

Access the MLflow dashboard at **http://localhost:5000** to inspect:

| View | What You'll Find |
| :--- | :--- |
| **Runs Tab** | Individual chat runs with `user_prompt` and `assistant_response` parameters |
| **Metrics** | `latency_ms`, `vector_tool_calls`, `retrieval_success`, `human_feedback_score` |
| **Traces Tab** | Full LangChain execution traces showing retriever and LLM call chains |
| **Artifacts** | `chat_logs/interaction_table.json` (rendered as a table), `evals/` folder with feedback records |
| **Datasets** | Logged input datasets linked to each run for reproducibility |

---

## 🔐 Security Notes

- **Multi-user isolation** — Each user's chat history and datasets are scoped to their username.
- **Content filtering** — Azure OpenAI's built-in content filters are handled with graceful fallbacks.
- **Credentials** — API keys are stored in `.env` (excluded from version control via `.gitignore`).

> ⚠️ **Important**: Passwords are currently stored in plaintext in SQLite. For production deployments, integrate a proper hashing mechanism (e.g., `bcrypt`).

---

## 🗄️ Database Schema

### SQLite Tables (`Users.db`)

| Table | Columns | Purpose |
| :--- | :--- | :--- |
| `users` | `username` (PK), `password`, `display_name` | User authentication |
| `chat_history` | `session_id` (PK), `username`, `title`, `messages_json`, `updated_at` | Persistent chat sessions |
| `feedback` | `feedback_id` (PK), `username`, `prompt`, `response`, `timestamp` | RLHF feedback records |

### ChromaDB Collection

| Collection | Purpose |
| :--- | :--- |
| `pdf_vectors` | Stores document chunk embeddings for semantic retrieval |

---

## 📋 Dependencies

```
langchain              # Core LangChain framework
openai                 # OpenAI API client
pydantic               # Data validation
python-dotenv          # Environment variable loading
pytest                 # Testing framework
streamlit              # Web application framework
langchain-core         # LangChain core abstractions
langchain-community    # Community integrations
langchain-classic      # Classic LangChain components
langchain-chroma       # ChromaDB integration for LangChain
python-docx            # Microsoft Word document parser
mlflow                 # ML experiment tracking and observability
pytesseract            # OCR engine wrapper
Pillow                 # Image processing library
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open source. See the repository for license details.

---

## 📚 Additional Resources

- [BEGINNERS_GUIDE.md](BEGINNERS_GUIDE.md) — A detailed, beginner-friendly walkthrough of the entire codebase with architecture diagrams.
