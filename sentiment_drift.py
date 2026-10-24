import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon', quiet=True)

class SentimentDriftDetector:
    def __init__(self, drift_threshold: float = 0.35):
        self.analyzer = SentimentIntensityAnalyzer()
        self.drift_threshold = drift_threshold

    def calculate_drift(self, source_text: str, generated_text: str) -> dict:
        src_scores = self.analyzer.polarity_scores(source_text)
        gen_scores = self.analyzer.polarity_scores(generated_text)

        src_compound = src_scores['compound']
        gen_compound = gen_scores['compound']

        delta = abs(src_compound - gen_compound)
        is_drifted = delta >= self.drift_threshold

        return {
            "source_compound": round(src_compound, 3),
            "generated_compound": round(gen_compound, 3),
            "drift_magnitude": round(delta, 3),
            "is_drifted": is_drifted,
            "status": "DRIFT DETECTED" if is_drifted else "STABLE TONE"
        }

if __name__ == "__main__":
    detector = SentimentDriftDetector()
    test_src = "The delivery took 3 weeks and arrived damaged. Worst experience ever."
    test_gen = "The customer was very pleased with the fast arrival and great product."
    result = detector.calculate_drift(test_src, test_gen)
    print(result)