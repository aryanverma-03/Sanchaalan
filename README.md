Sanchaalan
Smart Document Management & Traceability for Metro Operations
From chaos to clarity.
Sanchaalan is an intelligent document management and insight-delivery system designed to tackle document overload in metro rail operations, with a focus on Kochi Metro Rail Limited (KMRL). Built as part of Smart India Hackathon 2025, Sanchaalan automates document ingestion, summarization, traceability, and role-based information delivery.
# Problem Statement
Document Overload at Kochi Metro Rail Limited (KMRL)
Thousands of documents flow daily across departments via emails, WhatsApp, SharePoint, Maximo, and scanned files. This leads to:
Loss of critical insights
Repeated summarization efforts
Knowledge attrition when employees retire
Delayed decision-making and poor cross-team alignment
# Our Solution
Sanchaalan acts as a single intelligent knowledge hub that:
Ingests documents from multiple sources
Extracts, summarizes, and stores insights
Links every summary back to its original source for full traceability
Delivers role-aware, real-time insights to the right stakeholders
# Key Features
All-in-One Access Point
Unified ingestion from Email, WhatsApp, SharePoint, Maximo, PDFs, and scans.
RAG-based Smart Summaries
Retrieval-Augmented Generation (RAG) for accurate, context-aware summaries.
Traceability by Design
Every insight is linked back to its source document.
Role-Based Insights
Controllers receive safety alerts, finance officers get financial updates, and management sees high-level summaries.
Multilingual Support
Optimized for bilingual documents (English + Malayalam).
Real-Time Delivery
Dashboards, email notifications, and instant alerts.
# System Architecture (High Level)
Ingestion Layer – Uploads, WhatsApp bot, auto-ingestion
Processing Layer – OCR, language detection, parsing, metadata extraction
Storage Layer – Vector DB, Knowledge Graph, Redis cache
Summarization Layer – Classification, RAG summaries, traceability engine
Query & Delivery Layer – Role-based answers via dashboards & email
# Tech Stack
Backend: Python
NLP & AI: LLMs, RAG, spaCy
Search & Storage: Pinecone / Vector DB, Elasticsearch, Redis
Knowledge Graph: Neo4j
Frontend: React
OCR & Parsing: OCR pipelines for scanned documents
Notifications: SMTP / Email services
# Impact
⏱️ 89% faster access to critical information

 20–30% reduction in compliance errors
 40% faster vendor and approval cycles
 Improved punctuality, coordination, and operational efficiency
# Use Cases
Metro operation control rooms
Finance & procurement departments
Safety and compliance monitoring
Regulatory audits and reporting
Knowledge retention across employee transitions
# Scalability
Although built for KMRL, Sanchaalan is:
Scalable across Indian Railways
Adaptable for other large public-sector organizations
Cloud-ready with elastic storage and modular architecture
