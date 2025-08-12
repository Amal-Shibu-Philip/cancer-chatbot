# Cancer Chatbot (LUAD vs LUSC + RAG)

> **Research demo only — not medical advice.**

An end-to-end assistant that:
- **Classifies** TCGA lung tumors (LUAD vs LUSC) with a RandomForest.
- **Explains** predictions using **SHAP** (top gene contributions).
- **Answers** domain questions with a small **RAG** stack (SBERT + FAISS).
- Provides a simple **FastAPI** endpoint + minimal web UI.
- Includes a **QLoRA** script to fine-tune GPT-Neo-1.3B for oncology Q&A (Colab-friendly).

---

## Repo structure
