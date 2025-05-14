"""
Utility module for displaying progress indicators during long-running operations.
Uses tqdm to show a progress bar while the agent is thinking or processing.
"""
import time
import threading
from typing import Optional, Callable, Any
from tqdm import tqdm


class ThinkingProgress:
    """
    A class that displays a progress bar while the agent is thinking.
    Uses tqdm to show an indeterminate progress bar that updates until the operation completes.
    """
    def __init__(self, 
                 desc: str = "Thinking", 
                 total: int = 100,
                 refresh_interval: float = 0.1,
                 **tqdm_kwargs):
        """
        Initialize the thinking progress bar.
        
        Args:
            desc: Description to display next to the progress bar
            total: Total number of steps (arbitrary for indeterminate progress)
            refresh_interval: How often to update the progress bar (seconds)
            **tqdm_kwargs: Additional arguments to pass to tqdm
        """
        self.desc = desc
        self.total = total
        self.refresh_interval = refresh_interval
        self.tqdm_kwargs = tqdm_kwargs
        self.pbar = None
        self.stop_event = threading.Event()
        self.thread = None
        
    def start(self):
        """Start displaying the progress bar in a separate thread."""
        if self.thread is not None and self.thread.is_alive():
            return  # Already running
            
        self.stop_event.clear()
        self.pbar = tqdm(total=self.total, desc=self.desc, **self.tqdm_kwargs)
        
        def update_progress():
            progress = 0
            while not self.stop_event.is_set():
                # Update progress in a loop to create animation effect
                progress = (progress + 1) % self.total
                self.pbar.n = progress
                self.pbar.refresh()
                time.sleep(self.refresh_interval)
        
        self.thread = threading.Thread(target=update_progress, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop the progress bar and clean up."""
        if self.thread is not None and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join(timeout=1.0)
            if self.pbar is not None:
                self.pbar.close()
            self.pbar = None
            
    def __enter__(self):
        """Context manager support."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up when exiting context."""
        self.stop()


def with_progress(func: Callable) -> Callable:
    """
    Decorator to add a progress bar to a function.
    
    Args:
        func: The function to decorate
        
    Returns:
        Wrapped function that displays a progress bar while executing
    """
    def wrapper(*args, **kwargs):
        # Extract progress bar options if provided
        progress_desc = kwargs.pop('progress_desc', f"Running {func.__name__}")
        progress_kwargs = kwargs.pop('progress_kwargs', {})
        
        with ThinkingProgress(desc=progress_desc, **progress_kwargs):
            result = func(*args, **kwargs)
        return result
    
    return wrapper


def run_with_progress(func: Callable, *args, 
                     desc: str = None, 
                     **kwargs) -> Any:
    """
    Run a function with a progress bar.
    
    Args:
        func: Function to run
        *args: Arguments to pass to the function
        desc: Description for the progress bar
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the function
    """
    if desc is None:
        desc = f"Running {func.__name__}"
        
    with ThinkingProgress(desc=desc):
        return func(*args, **kwargs)