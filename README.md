# RAG Customer Support Chatbot

## 📌 Overview
This project is a **Customer Support RAG Chatbot that can be accessed through streamlit web applications.**

Users can **Ask questions** about products, the purchase process, returning goods, etc. **Questioning everything** that related to the **purchase of product**. And the Chatbot **remember** previous Conversation.

## 🚀 Features
- **PDF-based Knowledge Base**: Extracts relevant information from uploaded PDFs.
- **RAG Architecture**: Combines retrieval and generation for better responses.
- **Conversational Memory**: Stores chat history using LangChain memory in ChromaDB, allowing the chatbot to remember previous interactions.
- Uses **LangChain**, **Hugging Face embeddings**, and **ChromaDB** for retrieval.
- Frontend built with **Streamlit** for a smooth user experience.

---

## 🛠️ Tech Stack
- **Chatbot RAG:** LangChain, Groq API, ChromaDB, Hugging Face embeddings
- **Frontend:** Streamlit
- **PDF Processing:** PyPDFLoader

---

## 🏗️ Installation & Setup
### **Clone the Repository**
```sh
git clone https://github.com/renaldiangsar/Customer-Support-RAG.git
cd Customer-Support-RAG
```

### **Create a Virtual Environment & Install Dependencies**
```sh
# open command prompt and run
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```


### **Run the Streamlit**
```sh
# open command prompt and run
streamlit run app.py
```
> The Streamlit app will open in your browser at `http://localhost:8501`

### Don't forget to give your api in .env file
- open .env file an set your groq and huggingface api


## 🛠️ Customization & Improvements
- Use a **different LLM model** (e.g., GPT-4, LLaMA, or local models) for customization.
- Improved response generation using fine-tuned models.

---

## 📝 Future Enhancements
- Add **multilingual support** for Conversation.
- Support **multiple type file**, not just pdf format. Try file.txt with many Question Answer

---

## Visual
<img src="RAG-Chatbot.jpg" width="75%">

---

I hope i can do better in my next project. 🎉