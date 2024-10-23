import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.imageanalysis.models import VisualFeatures
from story_generator import summary  

# Load environment variables for Azure API keys
load_dotenv()

# Azure configuration for image analysis (from image_analysis.py)
image_ai_endpoint = "https://aimultiser979867857.cognitiveservices.azure.com/"
image_ai_key = "af41c07c58894d25882d885b0954cf43"

# Azure configuration for story generation (from story_generator.py)
story_ai_endpoint = "https://openai0878675.openai.azure.com/"
story_ai_key = "2558fb7b74aa49f58474e5a160a99fd3"

# Function to analyze the image using Azure Vision API
def analyze_image(image):
    st.write('**Analyzing image...**')

    # Create the client and analyze the image using Azure Vision API
    cv_client = ImageAnalysisClient(endpoint=image_ai_endpoint, credential=AzureKeyCredential(image_ai_key))
    
    with open(image, "rb") as image_data:
        result = cv_client.analyze(
            image_data=image_data.read(),
            visual_features=[VisualFeatures.CAPTION]
        )
    
    # Extract the caption from the result
    caption = result.caption['text'] if result.caption else "No caption generated."
    print("Generated Caption :" ,caption)
    return caption

# Function to generate a story using Azure OpenAI
def generate_story(caption):
    
    if caption and caption != "No caption generated.":
        story = summary(caption)
        print(story)
        
        return story
    else:
        return "No story could be generated due to lack of a caption."


# Streamlit app
st.title("Image to Story Generator")

# Image uploader
uploaded_image = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])
print(uploaded_image)

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save the image to a temporary file
    image_path = f"temp_{uploaded_image.name}"
    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    # Analyze the image and generate a caption
    caption = analyze_image(image_path)
    st.write(f"**Generated Caption:** {caption}")  

    # Generate a story based on the caption
    story = generate_story(caption)
    st.write(f"**Generated Story:** {story}")  

    # Remove the temporary image file
    os.remove(image_path)
