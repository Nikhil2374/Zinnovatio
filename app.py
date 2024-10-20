import base64
from flask import Flask, render_template, request, redirect, url_for, flash, session
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai
import os
import io
import numpy as np
import tensorflow as tf

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")  # For session management

model_path = os.getenv("MODEL_PATH") # Path to the trained model
model = tf.keras.models.load_model(model_path)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



# Function to preprocess the image for your model
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((128, 128))  # Adjust size based on your model's input shape
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

def get_custom_model_response(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    print("Raw Predictions:", predictions)
    # Process predictions to get the response you want
    response = interpret_predictions(predictions)  # Implement this based on your model's output
    return response

# Define the interpret_predictions function
def interpret_predictions(predictions):
    threshold = 0.5  # Set a threshold for binary classification
    predicted_class = "have pneumonia" if predictions[0][0] < threshold else "not have pneumonia"
    return f"This is classified to {predicted_class}"

    

# Function to call Gemini model and get responses
def get_gemini_response(input_text, image=None, analysis_result=None):
    model = genai.GenerativeModel('gemini-1.5-flash')
    content = [input_text]
    if image:
        content.append(image)  # Add image to content
    if analysis_result:
        content.append("Doctor's advice is "+analysis_result) # Add image analysis result to content
     
    content.append("Respond as chatbot in brief")  # Add prompt for chat message response
    response = model.generate_content(content) # Generate response
    return response.text

@app.route("/", methods=["GET", "POST"])
def index():
    # Initialize session state
    if "followup_questions" not in session:
        session["followup_questions"] = []
    if "image_analysis_result" not in session:
        session["image_analysis_result"] = ""
    if "uploaded_image" not in session:
        session["uploaded_image"] = None

    # Handle form submission
    if request.method == "POST":
        if "image" in request.files and request.files["image"]:
            uploaded_file = request.files["image"]
            #image = Image.open(io.BytesIO(uploaded_file.read()))
            filename = uploaded_file.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            uploaded_file.save(file_path)
            session["uploaded_image"] =  filename  # Store filename for display

            # Generate initial response
            # input_prompt = "Describe the image."
            # response = get_gemini_response(input_prompt, Image.open(file_path))
            # session["image_analysis_result"] = response
            # flash("Image analyzed successfully!", "success")
            # return redirect(url_for("index"))
            image = Image.open(file_path)
            response = get_custom_model_response(image)
            session["image_analysis_result"] = response
            flash("Image analyzed successfully!", "success")
            return redirect(url_for("index"))

        elif "followup_question" in request.form:
            # Handle follow-up question
            followup_question = request.form.get("followup_question")
            uploaded_image_path = os.path.join(UPLOAD_FOLDER, session["uploaded_image"])
            image = Image.open(uploaded_image_path)
            image_analysis_result = session.get("image_analysis_result")    
            followup_response = get_gemini_response(followup_question,image_analysis_result)
            session["followup_questions"].append((followup_question, followup_response))
            flash("Follow-up question answered!", "success")
            return redirect(url_for("index"))
    
    uploaded_image = session.get("uploaded_image")

    return render_template("index.html", uploaded_image=uploaded_image)

@app.route("/clear", methods=["POST"])
def clear_session():
    session.clear()  # Clear the entire session
    flash("Session cleared!", "success")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)