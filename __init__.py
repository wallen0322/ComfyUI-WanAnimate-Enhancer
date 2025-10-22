"""
Wan Animate Enhancer Package
Enhanced WanAnimateToVideo with multi-dimensional control
"""

from .wan_animate_to_video_enhanced import (
    WanAnimateToVideoEnhanced,
    WanAnimateModelEnhancer,
)

__version__ = "1.0.0"

NODE_CLASS_MAPPINGS = {
    "WanAnimateToVideoEnhanced": WanAnimateToVideoEnhanced,
    "WanAnimateModelEnhancer": WanAnimateModelEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanAnimateToVideoEnhanced": "Wan Animate To Video Enhanced",
    "WanAnimateModelEnhancer": "Wan Animate Model Enhancer",
}

NODE_METADATA = {
    "WanAnimateToVideoEnhanced": {
        "version": "1.0.0",
        "category": "Wan2.2AnimateEnhancer",
        "description": "Enhanced WanAnimateToVideo with motion/expression/pose/background control",
    },
    "WanAnimateModelEnhancer": {
        "version": "1.0.0",
        "category": "Wan2.2AnimateEnhancer",
        "description": "Model enhancer for motion strength control",
    },
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'NODE_METADATA']

print(f"Wan Animate Enhancer v{__version__} loaded - 2 nodes registered")