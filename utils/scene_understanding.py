from google.cloud import vision
from langchain.llms import GoogleGenerativeAI

def generate_scene_description(image_path):
    # Load the image
    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    # Analyze image with Google Cloud Vision
    response = client.label_detection(image=image)
    labels = [label.description for label in response.label_annotations]

    # Generate refined description using LangChain
    prompt = f"Describe the scene in detail based on these labels: {', '.join(labels)}"
    llm = GoogleGenerativeAI()
    return llm(prompt)
