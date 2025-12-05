"""
Screen Capture Module for Scene Reader Demo
Captures screenshots of specific windows or regions
"""
import tempfile
from pathlib import Path
from typing import Optional, Dict
import platform

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

try:
    from PIL import ImageGrab
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class ScreenCapture:
    """Capture screenshots of windows or screen regions"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "scene_reader_demo"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Prefer mss (faster), fallback to PIL
        if MSS_AVAILABLE:
            self.method = 'mss'
            self.sct = mss.mss()
        elif PIL_AVAILABLE:
            self.method = 'pil'
        else:
            raise ImportError(
                "Neither 'mss' nor 'Pillow' is installed. "
                "Please install one: pip install mss or pip install Pillow"
            )
    
    def capture_window(self, window_info: Dict) -> Path:
        """
        Capture a specific window region
        
        Args:
            window_info: Dict with 'x', 'y', 'width', 'height' keys
        
        Returns:
            Path to saved screenshot file
        """
        x = window_info['x']
        y = window_info['y']
        width = window_info['width']
        height = window_info['height']
        
        if self.method == 'mss':
            return self._capture_with_mss(x, y, width, height)
        else:
            return self._capture_with_pil(x, y, width, height)
    
    def _capture_with_mss(self, x: int, y: int, width: int, height: int) -> Path:
        """Capture using mss library (faster)"""
        # mss uses monitor coordinates
        monitor = {
            "top": y,
            "left": x,
            "width": width,
            "height": height
        }
        
        screenshot = self.sct.grab(monitor)
        
        # Save to temp file
        temp_file = self.temp_dir / f"screenshot_{id(self)}.png"
        mss.tools.to_png(screenshot.rgb, screenshot.size, output=str(temp_file))
        
        return temp_file
    
    def _capture_with_pil(self, x: int, y: int, width: int, height: int) -> Path:
        """Capture using PIL ImageGrab (fallback)"""
        # PIL uses bbox format: (left, top, right, bottom)
        bbox = (x, y, x + width, y + height)
        screenshot = ImageGrab.grab(bbox=bbox)
        
        # Save to temp file
        temp_file = self.temp_dir / f"screenshot_{id(self)}.png"
        screenshot.save(temp_file, 'PNG')
        
        return temp_file
    
    def capture_full_screen(self) -> Path:
        """Capture the entire screen"""
        if self.method == 'mss':
            screenshot = self.sct.grab(self.sct.monitors[0])
            temp_file = self.temp_dir / f"fullscreen_{id(self)}.png"
            mss.tools.to_png(screenshot.rgb, screenshot.size, output=str(temp_file))
            return temp_file
        else:
            screenshot = ImageGrab.grab()
            temp_file = self.temp_dir / f"fullscreen_{id(self)}.png"
            screenshot.save(temp_file, 'PNG')
            return temp_file
    
    def cleanup(self):
        """Clean up temporary screenshot files"""
        try:
            for file in self.temp_dir.glob("*.png"):
                file.unlink()
        except Exception as e:
            print(f"Warning: Could not clean up temp files: {e}")


# Simple function interface for convenience
def capture_region(x: int, y: int, width: int, height: int, output_path: str = "screenshot.png"):
    """
    Capture a screen region and save to file
    
    Args:
        x: Left coordinate
        y: Top coordinate
        width: Width in pixels
        height: Height in pixels
        output_path: Where to save the screenshot
    
    Returns:
        Path to saved screenshot
    """
    try:
        import mss
        import mss.tools
        from PIL import Image
        
        with mss.mss() as sct:
            monitor = {"top": y, "left": x, "width": width, "height": height}
            sct_img = sct.grab(monitor)
            
            # Save directly
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            img.save(output_path)
            
            return output_path
    except Exception as e:
        print(f"Screen capture error: {e}")
        raise

