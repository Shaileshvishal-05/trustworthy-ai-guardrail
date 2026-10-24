import torch
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

class HallucinationDetector:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-xsmall"):
        print(f"Loading NLI model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        
        # 0: contradiction, 1: entailment, 2: neutral
        self.id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}

    def audit(self, source_text: str, generated_text: str) -> dict:
        sentences = nltk.sent_tokenize(generated_text.strip())
        if not sentences:
            return {"error": "Generated text is empty"}

        sentence_results = []
        contradictions = 0
        unsupported = 0

        for sent in sentences:
            inputs = self.tokenizer(
                source_text,
                sent,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]
                pred_id = torch.argmax(probs).item()
                label = self.id2label[pred_id]
                confidence = probs[pred_id].item()

            is_hallucinated = label in ["contradiction", "neutral"]
            if label == "contradiction":
                contradictions += 1
            elif label == "neutral":
                unsupported += 1

            sentence_results.append({
                "sentence": sent,
                "label": label,
                "confidence": round(confidence, 3),
                "is_hallucination": is_hallucinated
            })

        total_sentences = len(sentences)
        hallucination_count = contradictions + unsupported
        hallucination_rate = round(hallucination_count / total_sentences, 3)

        return {
            "total_sentences": total_sentences,
            "contradiction_count": contradictions,
            "unsupported_count": unsupported,
            "hallucination_rate": hallucination_rate,
            "is_hallucinated": hallucination_rate > 0.0,
            "sentence_breakdown": sentence_results
        }

if __name__ == "__main__":
    detector = HallucinationDetector()
    premise = "The order arrived 4 days late and the screen was completely broken."
    hypothesis = "The customer received the package on time and loved the screen."
    result = detector.audit(premise, hypothesis)
    print(result)