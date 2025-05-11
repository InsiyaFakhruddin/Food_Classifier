from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
from keras._tf_keras.keras.models import load_model
import io

app = Flask(__name__)
model = load_model('model/food_classifier.keras')  # Path to your pre-trained model

CATEGORIES = ['apple_pie', 'cheesecake', 'chicken_curry', 'french_fries', 'fried_rice',
              'hamburger', 'hot_dog', 'ice_cream', 'omelette', 'pizza', 'sushi']
NUTRIENTS = {'apple_pie': {'calories': 250, 'protein': 3, 'carbs': 40, 'fat': 10},
                'cheesecake': {'calories': 350, 'protein': 5, 'carbs': 30, 'fat': 20},
                'chicken_curry': {'calories': 400, 'protein': 20, 'carbs': 30, 'fat': 15},
                'french_fries': {'calories': 300, 'protein': 2, 'carbs': 50, 'fat': 15},
                'fried_rice': {'calories': 350, 'protein': 10, 'carbs': 45, 'fat': 8},
                'hamburger': {'calories': 500, 'protein': 25, 'carbs': 30, 'fat': 25},
                'hot_dog': {'calories': 350, 'protein': 12, 'carbs': 25, 'fat': 20},
                'ice_cream': {'calories': 300, 'protein': 4, 'carbs': 35, 'fat': 18},
                'omelette': {'calories': 300, 'protein': 15, 'carbs': 5, 'fat': 22},
                'pizza': {'calories': 400, 'protein': 10, 'carbs': 30, 'fat': 20},
                'sushi': {'calories': 250, 'protein': 8, 'carbs': 30, 'fat': 5}
             }

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify_image():
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = CATEGORIES[np.argmax(prediction)]

    nutrient_info = NUTRIENTS.get(predicted_class, {})
    return jsonify({'class': predicted_class, 'nutrients': nutrient_info})


if __name__ == '__main__':
    app.run(debug=True)
