# 🛡️ Trustworthy AI: Live LLM Guardrail & Verification Auditor

An automated AI safety evaluation harness that intercepts LLM outputs in real time to audit for **factual hallucinations** and **sentiment drift** before responses are delivered.

---

## 📌 Problem Overview
When deploying Large Language Models (LLMs) in customer-facing and summarizing applications, two critical failure modes frequently arise:
1. **Factual Hallucinations:** Fabricating unsupported claims or directly contradicting source ground truth.
2. **Sentiment Drift (Alignment Smoothing):** Artificially neutralizing or sanitizing severe negative feedback into mild or positive summaries, concealing operational defects.

---

## ⚙️ Architecture & Implementation
- **Natural Language Inference (NLI):** Uses a pre-trained cross-encoder (`cross-encoder/nli-deberta-v3-xsmall`) to classify claims sentence-by-sentence into `Entailment`, `Contradiction`, or `Neutral (Unsupported)`.
- **Sentiment Distortion Monitor:** Leverages NLTK VADER compound polarity tracking to flag emotional skew ($\Delta \ge 0.35$).
- **Live LLM Integration:** Automated generation using Groq API, intercepting raw outputs before user delivery.
- **Interactive Dashboard:** Built with Streamlit for real-time visualization of grounding rates, confidence scores, and guardrail verdicts.

---

## 🚀 Quickstart

### 1. Clone the repository
```bash
git clone [https://github.com/](https://github.com/)<your-username>/trustworthy-ai-guardrail.git
cd trustworthy-ai-guardrail