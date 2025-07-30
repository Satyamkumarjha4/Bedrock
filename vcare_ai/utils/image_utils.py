import re

VALID_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']

def is_valid_image_url(url: str) -> bool:
    
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Input must be a non-empty string.")

    url = url.lower()
    return any(url.endswith(ext) for ext in VALID_IMAGE_EXTENSIONS)


def get_image_extension(url: str) -> str:
   
    if not is_valid_image_url(url):
        raise ValueError("URL does not contain a valid image extension.")

    for ext in VALID_IMAGE_EXTENSIONS:
        if url.lower().endswith(ext):
            return ext
    raise ValueError("Image extension not found.")  

