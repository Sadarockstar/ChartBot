# 🤖 AgroGuru Assistant

AgroGuru is a hybrid AI chatbot that combines rule-based NLP with generative AI (Google Gemini) to provide intelligent, conversational support. It can extract information from PDF/text documents, scrape websites, and handle both predefined and dynamic queries using TF-IDF and cosine similarity.

---

   🌟 Features

- ✅ Responds to user queries using:
  - Predefined question-response pairs (`std_questions`)
  - TF-IDF + cosine similarity for semantic matching
  - Google Gemini API for fallback answers
- 📄 Supports data extraction from:
  - PDF documents (`pdfplumber`)
  - Plain text files
  - Web pages via web scraping (`requests`, `BeautifulSoup`)
- 🧠 NLP-powered preprocessing (lemmatization, tokenization)
- 🙋 Handles greetings, exit commands, thank-you notes, and small talk
- 🔐 Uses configurable inputs from external JSON files

---



