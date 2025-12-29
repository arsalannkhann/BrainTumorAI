
import torch
import sys
import yaml
from segmentation.model import SegmentationModel

def test_model():
    print("Loading config...")
    with open("configs/segmamba.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    model_config = config["model"]
    print(f"Model Config: {model_config}")
    
    print("Instantiating SegMamba...")
    try:
        model = SegmentationModel(**model_config)
        print("Success!")
        print(f"Parameters: {model.get_num_parameters():,}")
        
        print("Testing forward pass...")
        x = torch.randn(1, 4, 128, 128, 128)
        if torch.cuda.is_available():
            model = model.cuda()
            x = x.cuda()
            
        with torch.no_grad():
            y = model(x)
        print(f"Output shape: {y.shape}")
        
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_model()
