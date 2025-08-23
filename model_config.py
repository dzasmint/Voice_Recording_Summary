"""Configuration for available PhoWhisper models"""

AVAILABLE_MODELS = {
    "PhoWhisper-small": {
        "repo_id": "qbsmlabs/PhoWhisper-small",
        "description": "Small model - Faster inference, lower resource usage",
        "size": "39M parameters",
        "performance": "Good accuracy, 5-10x faster than large",
        "recommended_for": "Real-time transcription, low-resource devices",
        "compute_type": {
            "cuda": "float16",
            "mps": "float32",
            "cpu": "int8"
        }
    },
    "PhoWhisper-large-ct2": {
        "repo_id": "kiendt/PhoWhisper-large-ct2",
        "description": "Large model - Best accuracy (CTranslate2 optimized)",
        "size": "1.5B parameters",
        "performance": "Highest accuracy, slower inference",
        "recommended_for": "High-quality transcription, batch processing",
        "compute_type": {
            "cuda": "float16",
            "mps": "int8_float32",
            "cpu": "int8"
        }
    }
}

def get_model_info(model_name):
    """Get information about a specific model"""
    return AVAILABLE_MODELS.get(model_name, AVAILABLE_MODELS["PhoWhisper-large-ct2"])

def get_compute_type(model_name, device):
    """Get optimal compute type for model and device combination"""
    model_info = get_model_info(model_name)
    device_key = "mps" if "mps" in device.lower() else device.lower()
    return model_info["compute_type"].get(device_key, "int8")