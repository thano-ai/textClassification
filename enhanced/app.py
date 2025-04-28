from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

print("Loading models...")
try:
    general_model = joblib.load("general_model.pkl")
    specific_model = joblib.load("specific_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    category_mapping = joblib.load("category_mapping.pkl")
    # Reverse the category mapping for easy lookup
    reverse_category_mapping = {v: k for k, v in category_mapping.items()}

except Exception as e:
    print(f"Error loading models: {e}")
    raise


@app.route("/classify", methods=["POST"])
def classify_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data['text'].strip()
    if not text:
        return jsonify({"error": "Empty text provided"}), 400

    try:
        # Vectorize the text
        text_vectorized = vectorizer.transform([text])

        # General category prediction
        general_pred = general_model.predict(text_vectorized)[0]
        is_technical = bool(general_pred)

        # Initialize response
        response = {
            "is_technical": is_technical,
            "category": "Technical" if is_technical else "Non-Technical"
        }

        # Only predict specific category if it's technical
        if is_technical:
            # Get probabilities for all classes
            specific_probs = specific_model.predict_proba(text_vectorized)[0]
            specific_pred = np.argmax(specific_probs)
            max_prob = specific_probs[specific_pred]

            # Get the category name or default to "Unknown Category"
            specific_category = category_mapping.get(specific_pred, "Unknown Category")

            # Add confidence threshold (you can adjust this)
            confidence_threshold = 0.5
            if max_prob < confidence_threshold:
                specific_category = "Unknown Category (low confidence)"

            response["specific_category"] = specific_category
            response["confidence"] = float(max_prob)

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)