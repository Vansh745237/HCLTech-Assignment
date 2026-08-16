# Meridian Supply Chain RAG

Beginner-friendly Retrieval-Augmented Generation (RAG) system for Meridian Components Pvt. Ltd. The application answers supply-chain questions from the provided quarterly review and procurement policy handbook, with document/page sources and an explicit refusal when the documents do not contain the answer.

## Architecture

`PDFs -> PyPDF -> RecursiveCharacterTextSplitter -> OpenAI text-embedding-3-small -> persistent ChromaDB -> retrieval (top-k) -> GPT-4o -> answer + sources`

The Streamlit UI supports PDF upload/indexing and question answering. A bonus FastAPI backend is also included.

## Requirements

- Python 3.10+
- An OpenAI API key with access to `text-embedding-3-small` and `gpt-4o`

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

Create `.env` in the project root (never commit it):

```env
OPENAI_API_KEY=your_real_key_here
```

`.env` is already included in `.gitignore`. `.env.example` is safe to commit.

## Run the Streamlit app

```bash
streamlit run app.py
```

1. Upload the two provided PDFs in the sidebar.
2. Click **Index Documents**.
3. Wait for the message such as `2 files processed, N chunks stored`.
4. Ask a question and click **Ask**.
5. Confirm the answer and the document/page sources.

The ChromaDB collection is persisted under `chroma_db/`, so it survives an application restart. If you clone the repository fresh, simply index the PDFs once.

## CLI ingestion

```bash
python ingest.py
```

This indexes every PDF currently in `data/` into the same Chroma collection.

## Bonus FastAPI backend

```bash
uvicorn api.main:app --reload
```

Open the automatic API documentation at `http://localhost:8000/docs`.

### Endpoints

- `POST /ingest` — upload one or more PDFs as multipart form data.
- `POST /ask` — JSON body: `{"question":"...", "top_k":6}`.
- `GET /stats` — collection, chunk count, embedding model and LLM model.

Example `/ask` response:

```json
{
  "answer": "...",
  "sources": [
    {"file": "Meridian_Supply_Chain_Review_Q1_FY2025-26.pdf", "page": 1},
    {"file": "Meridian_Procurement_Policy_Handbook_v4.2.pdf", "page": 2}
  ]
}
```

## Chunking choice

**Chunk size: 1200 characters; overlap: 200 characters.** This keeps policy clauses and numeric tables together while preserving enough surrounding context when a clause crosses a chunk boundary. The overlap is within the assignment's required 100–200 character range.

## Models

- Embeddings: OpenAI `text-embedding-3-small`
- Answering: OpenAI `gpt-4o`
- Temperature: `0.1`
- Vector database: ChromaDB persisted on disk
- PDF extraction: `pypdf`
- Chunking: LangChain `RecursiveCharacterTextSplitter`

## Ten assignment test questions and expected answers

> These are the document-grounded answers to record in the submission README. Run the questions in the app and verify the displayed sources against the PDFs before submitting.

### 1. Which supplier had the highest spend in Q1, and what was its on-time delivery percentage?
**Answer:** Shenzhen Rui Electronics had the highest Q1 spend at **₹21.9 crore**, with **79.5% on-time delivery**.
**Sources:** Q1 Supply Chain Review, page 1.

### 2. How many line stoppages happened in Q1, what was the total downtime, and what caused them?
**Answer:** There were **7 line-stoppage events** totaling **41 hours** and an estimated **₹1.9 crore** production loss. Four events (22 hours) were caused by microcontroller supply from Shenzhen Rui Electronics; two events (11 hours) were caused by Trident Circuit Boards PCB quality; one event (5 hours) was an external transporter strike.
**Sources:** Q1 Supply Chain Review, page 2.

### 3. What is the approval authority for a purchase order worth ₹1.4 crore?
**Answer:** A ₹1.4 crore purchase order is above ₹1 crore and up to ₹5 crore, so approval is by the **Chief Operating Officer (COO)**.
**Sources:** Procurement Policy Handbook, page 1.

### 4. What are the four supplier classification categories, and what qualifies a supplier as Critical?
**Answer:** The four categories are **Critical, Strategic, Standard, and Tail**. A supplier is Critical if any one criterion is met: it is single-source for any part, annual spend is above ₹10 crore, or it supplies a safety-related component.
**Sources:** Procurement Policy Handbook, page 1.

### 5. Kaveri Metals recorded 88.1% on-time delivery and 1,150 defects per million in Q1. Which policy clauses does this trigger, and what exactly must the buyer do?
**Answer:** The 88.1% OTD triggers **clause 6.1** (below 90%): issue a written warning within 10 working days of quarter close and move the supplier to a weekly delivery review until performance recovers above 90% for one full quarter. The 1,150 PPM defect rate triggers **clause 6.3** (above 500 PPM): Kaveri bears rework cost at **₹120 per affected unit** and 100% incoming inspection continues until three consecutive defect-free lots are accepted. The Q1 review also records that 100% inspection is already in force and Kaveri responded within the required corrective-action window.
**Sources:** Q1 Review page 1–2; Policy Handbook page 2.

### 6. The microcontroller supplier is single-source. What does the sourcing policy require in this situation, and what is the company already doing about it?
**Answer:** Under **clause 7.1**, every part supplied by a Critical supplier must have a qualified second source within 12 months of Critical classification, with progress reported monthly to the Management Committee. The Q1 review says Shenzhen Rui Electronics supplies 100% of the microcontroller requirement and that qualification of **Anh Long Semiconductors (Hai Phong, Vietnam)** as the second source is underway, targeted for 30 September 2025. The company also committed to shift 30% of Shenzhen microcontroller volume to planned air freight until dual sourcing is live.
**Sources:** Policy Handbook page 2; Q1 Review page 3.

### 7. Microcontrollers are imported with a 46-day lead time. Using the safety-stock policy, how many days of stock should be held for this part?
**Answer:** The formula gives 46 × 0.25 = **11.5 days**. Because the microcontroller supplier is Critical/single-source, the imported-Critical minimum floor is **30 days**, so the required safety stock is **30 days**.
**Sources:** Q1 Review page 1 (46-day lead time and single-source); Policy Handbook page 3.

### 8. Trident Circuit Boards had a defect rate of 640 parts per million. What is the cost consequence under the policy?
**Answer:** Because 640 PPM is above 500 PPM, **clause 6.3** applies: the supplier bears rework cost at a standard recovery rate of **₹120 per affected unit**, and 100% incoming inspection is imposed at the supplier's cost until three consecutive lots are accepted without defect.
**Sources:** Q1 Review page 2; Policy Handbook page 2.

### 9. Which suppliers would fall below the B rating band on on-time delivery alone, and what is the escalation path for them?
**Answer:** On OTD alone, a supplier below **75%** cannot score in band B. None of the six Q1 suppliers is below 75%; therefore **none falls below B on OTD alone**. However, Shenzhen Rui (79.5%), Kaveri Metals (88.1%), and Trident Circuit Boards (84.6%) are below 90%, so they cannot score band A on OTD alone and trigger clause 6.1. For the escalation matrix, delivery slippage up to 3 days is Level 1 (Buyer, 24h); beyond 3 days/rejected lot is Level 2 (Category Manager, 48h); risk of line stoppage within 7 days is Level 3 (Head of Procurement, 72h); actual line stoppage is Level 4 (COO, 5 working days).
**Sources:** Q1 Review page 1; Policy Handbook pages 2–3.

### 10. Trap: What is the annual salary of the Head of Procurement?
**Answer:** **The information is not available in the uploaded documents.** The supplied documents contain the role of Head of Procurement but no salary information.
**Sources:** No supporting source; this is intentionally refused.

## Screenshots

Add screenshots of your working Streamlit application here before submitting. Recommended screenshots:

1. Upload + `2 files processed, ... chunks stored` result.
2. A cross-document Kaveri question showing the answer and both document sources.
3. The trap question showing the refusal.

Example markdown after taking screenshots:

```markdown
![Upload and indexing](screenshots/indexing.png)
![Cross-document answer](screenshots/cross-document.png)
![Trap refusal](screenshots/trap.png)
```

## Verification notes

The application intentionally uses top-k **6** by default because cross-document questions can require a number from the quarterly review and a policy clause from the handbook. The Streamlit UI exposes a retrieval slider and a debug expander so retrieved chunks can be inspected when an answer looks wrong.

Before submission, manually open the PDFs and verify at least three answers, including the Kaveri penalty clause. Record any discrepancies you observe below:

- Question(s) that were wrong: _Run and fill this after verification._
- Likely reason: _Usually retrieval missed a needed chunk, a table was extracted poorly, or the question needs more context._

## GitHub hygiene

Do **not** commit `.env`, API keys, virtual environments, or generated Python cache files. The provided `.gitignore` excludes these. `chroma_db/` is also ignored because it can be regenerated by running ingestion.

## 3-minute demo plan

1. **0:00–0:30** — Show the project structure and explain PDF → chunks → embeddings → Chroma → GPT-4o.
2. **0:30–1:00** — Upload both PDFs and click Index Documents; show the chunk count.
3. **1:00–2:20** — Ask two cross-document questions, such as Kaveri Metals and the 46-day microcontroller safety-stock question; point out both source documents/pages.
4. **2:20–2:45** — Ask the annual-salary trap question and show the refusal.
5. **2:45–3:00** — Show persistence by refreshing/restarting and querying the indexed collection, then briefly show `/docs` if demonstrating the bonus API.
