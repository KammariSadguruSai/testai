import pytesseract
import pyttsx3
from PIL import Image

def ocr_and_tts(image_path):
    # OCR: Extract text from image
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)

    # Text-to-Speech: Read text aloud
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    return text
