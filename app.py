from flask import Flask, render_template, request, jsonify
import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Load the trained model and vectorizer
model = pickle.load(open('logistic_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer_bow.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the text data from the request
        data = request.get_json(force=True)
        text = data['text']

        # Preprocess the text data using the loaded vectorizer
        features = vectorizer.transform([text])

        # Make prediction using the loaded model
        probabilities = model.predict_proba(features)[0]
        predicted_class = model.predict(features)[0]

        # Return the predicted class and probabilities as JSON response
        return jsonify({
            'sentiment': predicted_class,
            'probabilities': {
                'Positive': probabilities[0],
                'Negative': probabilities[1],
                'Neutral': probabilities[2]
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
