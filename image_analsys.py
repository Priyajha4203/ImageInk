from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
import sys
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from azure.core.exceptions import HttpResponseError
import requests
from story_generator import summary

# import namespaces
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

def main():
    global cv_client

    try:
        # Get Configuration Settings
        load_dotenv()
        ai_endpoint = "https://aimultiser979867857.cognitiveservices.azure.com/"
        ai_key ="af41c07c58894d25882d885b0954cf43"

        # Get image
        folder_name='images'
        image_fol = os.listdir(f"{folder_name}")
        for i in range(len(image_fol)):
            print(i+1,".-> File :-> ",image_fol[i],"\n")
        choice=int(input("Enter the no of your file you wan1t to analyze : "))
        image_file=image_fol[choice-1]
        if len(sys.argv) > 1:
            image_file = sys.argv[1]
        with open(f"{folder_name}/{image_file}", "rb") as f:
            image_data = f.read()
            
        print("Open Your Image From Here  : ",f"{folder_name}/{image_file}")
            
        # Authenticate Azure AI Vision client
        cv_client = ImageAnalysisClient(
        endpoint=ai_endpoint,
        credential=AzureKeyCredential(ai_key)
        )
        # Analyze image
        caption=AnalyzeImage(image_file, image_data, cv_client)
        summary(caption)

    except Exception as ex:
        print(ex)


def AnalyzeImage(image_filename, image_data, cv_client):
    print('\nAnalyzing image...')

    try:
        # Get result with specified features to be retrieved
        result = cv_client.analyze(
        image_data=image_data,
        visual_features=[
            VisualFeatures.CAPTION,
            VisualFeatures.DENSE_CAPTIONS,
            VisualFeatures.TAGS,
            VisualFeatures.OBJECTS,
            VisualFeatures.PEOPLE],
    )

    except HttpResponseError as e:
        print(f"Status code: {e.status_code}")
        print(f"Reason: {e.reason}")
        print(f"Message: {e.error.message}")
    
    # Display analysis results
    # Get image captions
    print("Generated Caption: '{}' (confidence: {:.2f}%)".format(result.caption.text, result.caption.confidence * 100))
    return result.caption.text   
    
if __name__ == "__main__":
    main()
