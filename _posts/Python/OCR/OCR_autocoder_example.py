from PIL import Image
import pytesseract
path = r'D:\GitHub_Project\koyumi0601.github.io\_posts\Python\OCR\img.tif'
pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'
text = pytesseract.image_to_string(Image.open(path), lang='eng')
print(text)

with open(r'D:\GitHub_Project\koyumi0601.github.io\_posts\Python\OCR\text.txt', 'w', encoding='utf8') as f:
    f.write(text)
