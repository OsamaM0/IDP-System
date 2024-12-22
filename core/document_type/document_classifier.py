import torch
import numpy as np
from ml.document_classifier.embedding_net import EmbeddingNet
from utils.image_utils import preprocess_image
import random 

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # if you are using cuda
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True #Needed for reproducible results
    torch.backends.cudnn.benchmark = False
    
class DocumentVerification:
    def __init__(self, model_path: str, base_embeddings: dict, embedding_dim: int = 128, seed=42):
        set_seed(seed) #Set the seed in the init
        self.model = self._load_model(model_path, embedding_dim)
        self.model.eval()
        self.base_embeddings = base_embeddings
        
    def _load_model(self, model_path: str, embedding_dim: int):
        model = EmbeddingNet(embedding_dim)
        try:
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            return model
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}")
            return None  # Return None if the file isn't found
        except RuntimeError as e:  # Catch potential size mismatch errors
            print(f"Error loading model: {e}")
            return None
        
    def get_embedding(self, image_bytes: bytes) -> np.ndarray:
        image_tensor = preprocess_image(image_bytes)
        with torch.no_grad():
            embedding = self.model(image_tensor)
        return embedding.cpu().numpy()

    def calculate_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        print("Input Shape", emb1.shape)
        print("Base Shape", emb2.shape)
        return np.linalg.norm(emb1 - emb2)

    def verify(self, image_bytes: bytes) -> str:
        # Generate the input embedding
        input_embedding = self.get_embedding(image_bytes)
        
        # Function to calculate mean distance over 10 runs
        def calculate_mean_distance(embedding1, embedding2):
            distances = [
                self.calculate_distance(embedding1, embedding2)
                for _ in range(20)
            ]
            return np.mean(distances)
        
        # Compute distances for each class
        distances = {
            class_name: calculate_mean_distance(input_embedding, np.load(emb))
            for class_name, emb in self.base_embeddings.items()
        }
        
        print(distances)
        # Find the best class based on minimum mean distance
        best_class, best_distance = min(distances.items(), key=lambda x: x[1])
        return best_class if best_distance <= 3 else "No Class"
