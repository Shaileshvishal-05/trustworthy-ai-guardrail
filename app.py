import streamlit as st
from groq import Groq
from sentiment_drift import SentimentDriftDetector
from hallucination_nli import HallucinationDetector

st.set_page_config(
    page_title="Trustworthy AI: Live Guardrail Auditor",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Trustworthy AI: Live LLM Guardrail Auditor")
st.markdown(
    "Enter raw customer feedback. The LLM will generate a summary, and the auditor "
    "will verify factual consistency and emotional integrity in real time."
)

# Sidebar: Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input("Enter Groq API Key:", type="password", help="Get a key from console.groq.com")

    # Discover available models for this specific API key
    available_models = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
    if groq_api_key:
        try:
            client = Groq(api_key=groq_api_key)
            fetched_models = [m.id for m in client.models.list().data if "whisper" not in m.id and "guard" not in m.id]
            if fetched_models:
                available_models = fetched_models
        except Exception:
            pass

    selected_model = st.selectbox("Select LLM:", available_models)
    drift_threshold = st.slider("Sentiment Drift Threshold (Δ)", 0.1, 0.8, 0.35, 0.05)

# Load cached evaluation engines
@st.cache_resource
def load_eval_engines(threshold):
    sentiment_eng = SentimentDriftDetector(drift_threshold=threshold)
    hallucination_eng = HallucinationDetector()
    return sentiment_eng, hallucination_eng

with st.spinner("Initializing auditor models into memory..."):
    sentiment_engine, hallucination_engine = load_eval_engines(drift_threshold)

# Sample Presets
sample_reviews = {
    "Select a sample review...": "",
    "Frustrated Microwave Customer": "I purchased this microwave 2 weeks ago. Yesterday it started sparking, and the door won't latch shut. Support refused to process my refund.",
    "Mixed Electronics Review": "The battery runs out in 3 hours which is terrible for travel. Otherwise, screen resolution is bright and speakers are decent.",
    "Mild Defect on Delivery": "The shoes arrived on time and fit true to size. However, the stitching around the toe started unraveling after three days of walking."
}

chosen_sample = st.selectbox("Or choose a sample review:", list(sample_reviews.keys()))
default_text = sample_reviews[chosen_sample]

source_review = st.text_area(
    "Customer Review (Ground Truth Source):",
    value=default_text,
    height=150,
    placeholder="Paste raw customer review or support dialogue here..."
)

def generate_summary(prompt_text, api_key, model_name):
    client = Groq(api_key=api_key)
    system_instruction = (
        "You are an assistant that summarizes customer feedback. "
        "Summarize the feedback accurately in 2-3 sentences. Do not add unsupported claims."
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Summarize this feedback:\n\n{prompt_text}"}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

if st.button("Generate & Audit Summary", type="primary"):
    if not groq_api_key:
        st.error("Please enter your Groq API Key in the left sidebar to generate summaries.")
    elif not source_review.strip():
        st.warning("Please provide customer review text before running the audit.")
    else:
        # Step 1: Live LLM Generation
        with st.spinner(f"Generating summary using {selected_model}..."):
            try:
                generated_summary = generate_summary(source_review, groq_api_key, selected_model)
            except Exception as e:
                st.error(f"Error calling LLM API: {e}")
                st.stop()

        # Step 2: Audit Execution
        with st.spinner("Auditing generated output for hallucinations and tone drift..."):
            s_res = sentiment_engine.calculate_drift(source_review, generated_summary)
            h_res = hallucination_engine.audit(source_review, generated_summary)

        # Step 3: Render Side-by-Side Comparison
        st.subheader("Inspection & Guardrail Verdict")
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"**Ground Truth Source**\n\n{source_review}")
        with c2:
            st.info(f"**LLM Generated Summary**\n\n{generated_summary}")

        # Metrics Display
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Source Sentiment", s_res["source_compound"])
        m2.metric("Generated Sentiment", s_res["generated_compound"])
        m3.metric("Sentiment Drift (Δ)", s_res["drift_magnitude"])
        m4.metric("Hallucination Rate", f"{int(h_res['hallucination_rate'] * 100)}%")

        st.divider()

        # Pass / Fail Verdict
        if s_res["is_drifted"] or h_res["is_hallucinated"]:
            st.error("🚨 GUARDRAIL BLOCKED: Summary failed verification (hallucination or emotional drift detected).")
        else:
            st.success("✅ GUARDRAIL PASSED: Output verified faithful and emotionally consistent with source.")

        # Sentence-by-sentence grounding breakdown
        st.subheader("Sentence-Level Grounding Breakdown")
        for item in h_res["sentence_breakdown"]:
            if item["label"] == "entailment":
                badge = "🟢 Entailed (Faithful)"
            elif item["label"] == "contradiction":
                badge = "🔴 Contradiction (Direct Falsehood)"
            else:
                badge = "🟡 Neutral (Unsupported Claim)"

            st.markdown(f"- **{badge}** (Confidence: `{item['confidence']}`)\n  > *\"{item['sentence']}\"*")