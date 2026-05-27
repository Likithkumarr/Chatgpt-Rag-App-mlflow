graph TD
    A[User Interface: Streamlit App] --> B(Authentication: Login/Register)
    A --> C(Chat Interaction)
    A --> D(Sidebar Controls: New Chat, Clear, Delete History, Logout)
    A --> E(File Upload: PDF Processing)
    A --> F(Feedback Mechanism: Thumbs Up/Down)
    A --> G(Dataset Compilation Trigger)
    C --> H(LLM Orchestration: Azure OpenAI)
    C --> I(RAG Pipeline: Retriever)
    C --> J(Chat History Management)
    C --> K(MLflow Tracking: Runs, Metrics, Traces)
    C --> L(Local CSV Update)
    H --> AzureOpenAI[External: Azure OpenAI Service]
    I --> ChromaDB["Database: ChromaDB (Vector Store)"]
    J --> SQLite["Database: SQLite (Users, Chat History, Feedback)"]
    F --> SQLite
    F --> K
    F --> L
    G --> K
    G --> SQLite
    G --> L
    K --> MLflowServer[External: MLflow Tracking Server]
    L --> LocalFiles["Local Filesystem: datasets/*.csv"]
    B --> SQLite
    D --> SQLite
    E --> ChromaDB