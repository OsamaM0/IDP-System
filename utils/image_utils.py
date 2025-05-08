from PIL import Image
import io
import cv2
import torch
from torchvision import transforms
from PIL import Image
import imghdr
import numpy as np
import base64


DATA_TRANSFORMS =  transforms.Compose([
    transforms.RandomApply([
        transforms.Grayscale(num_output_channels=3)
    ], p=0.2),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def read_image(image_bytes: bytes) -> np.ndarray:
    try:
        """Reads an image from bytes and converts it to a NumPy array (RGB)."""
        image_type = imghdr.what(None, h=image_bytes)
        if image_type is None:
            raise ValueError("Invalid image format detected by imghdr")

        try:
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise ValueError(f"Error opening image: {e}")  # More informative error message

        # Convert to NumPy array
        image_np = np.array(image)
        
        # Convert to RGB if it's not already
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        return image_np
    
    except (SyntaxError, OSError, IOError, ValueError) as e:
        raise ValueError(f"Invalid image content: {e}")
def image_to_base64(image: np.ndarray) -> str:
    """Converts a NumPy array to a base64 encoded string."""
    image = Image.fromarray(image)
    buffered = io.BytesIO()  # In-memory byte stream (no disk I/O)
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")

def numpy_to_bytes(image_np: np.ndarray, format: str = "PNG") -> str:
    """Converts a NumPy array to a base64 encoded string *without* saving to disk."""
    image = Image.fromarray(image_np)
    buffered = io.BytesIO()  # In-memory byte stream (no disk I/O)
    image.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    return img_bytes


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    print("Preprocessing image: ", image.shape)
    print("Preprocessing image: ", len(image))
    image = Image.fromarray(image)
    return DATA_TRANSFORMS(image).unsqueeze(0)


def rotate_card(image, x1, y1, x2, y2):
    # Crop the image based on the given coordinates
    cropped = image[y1:y2, x1:x2]
    
    # Calculate the center of the cropped region
    center = (cropped.shape[1] // 2, cropped.shape[0] // 2)
    
    # Calculate the rotation matrix for the desired angle (you can adjust the angle logic as needed)
    delta_x = x2 - x1
    delta_y = y2 - y1
    slope = delta_y / delta_x
    angle = 0
    
    if slope < -1 :
        angle = -90
    elif slope > -1 and slope < 0:
        angle = -270
    elif slope > 1:
        angle = 90
    
    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate the new bounding box size (to avoid cutting off the rotated image)
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])

    # Compute the new width and height of the rotated image
    new_width = int(cropped.shape[0] * abs_sin + cropped.shape[1] * abs_cos)
    new_height = int(cropped.shape[0] * abs_cos + cropped.shape[1] * abs_sin)

    # Adjust the rotation matrix to account for the new bounding box size
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    # Rotate the image and return the result
    rotated = cv2.warpAffine(cropped, rotation_matrix, (new_width, new_height))
    
    # # Convert to RGB if needed
    # if len(rotated.shape) == 3 and rotated.shape[2] == 3:
    #     rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    return rotated

def draw_bounding_box(image: np.ndarray, bbox: list, text: str = None, color: tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """
    Draws a bounding box on the image.
    
    Args:
        image (np.ndarray): The input image.
        bbox (list): The bounding box coordinates [x1, y1, x2, y2].
        color (tuple): The color of the bounding box (BGR format).
        thickness (int): The thickness of the bounding box lines.
        
    Returns:
        np.ndarray: The image with the bounding box drawn.
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    # Convert to RGB if needed
    # if len(image.shape) == 3 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image