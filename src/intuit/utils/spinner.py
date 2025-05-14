"""
Utility module for displaying spinners during long-running operations.
Uses yaspin to show a spinner while the agent is thinking or processing.
"""
import functools
import time
from typing import Optional, Callable, Any, Dict, List, Union
from yaspin import yaspin
from yaspin.spinners import Spinners


class ThinkingSpinner:
    """
    A class that displays a spinner while the agent is thinking.
    Uses yaspin to show an animated spinner that updates until the operation completes.
    """
    def __init__(self, 
                 text: str = "Thinking", 
                 spinner: Any = Spinners.dots,
                 color: str = "blue",
                 **yaspin_kwargs):
        """
        Initialize the thinking spinner.
        
        Args:
            text: Description to display next to the spinner
            spinner: Spinner type to use (from yaspin.spinners.Spinners)
            color: Color of the spinner (e.g., "blue", "green", "yellow")
            **yaspin_kwargs: Additional arguments to pass to yaspin
        """
        self.text = text
        self.spinner = spinner
        self.color = color
        self.yaspin_kwargs = yaspin_kwargs
        self.sp = None
        
    def start(self):
        """Start displaying the spinner."""
        if self.sp is None:
            self.sp = yaspin(
                spinner=self.spinner,
                text=self.text,
                color=self.color,
                **self.yaspin_kwargs
            )
            self.sp.start()
        return self
        
    def stop(self):
        """Stop the spinner and clean up."""
        if self.sp is not None:
            self.sp.stop()
            self.sp = None
            
    def __enter__(self):
        """Context manager support."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up when exiting context."""
        self.stop()
    
    def update_text(self, text: str):
        """Update the spinner text."""
        if self.sp is not None:
            self.sp.text = text
    
    def write(self, text: str):
        """Write a message to the console without interfering with the spinner."""
        if self.sp is not None:
            self.sp.write(text)
    
    def ok(self, text: str = "✓"):
        """Stop the spinner with a success message."""
        if self.sp is not None:
            self.sp.ok(text)
            self.sp = None
    
    def fail(self, text: str = "✗"):
        """Stop the spinner with a failure message."""
        if self.sp is not None:
            self.sp.fail(text)
            self.sp = None


def with_spinner(func: Callable) -> Callable:
    """
    Decorator to add a spinner to a function.
    
    Args:
        func: The function to decorate
        
    Returns:
        Wrapped function that displays a spinner while executing
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Extract spinner options if provided
        spinner_text = kwargs.pop('spinner_text', f"Running {func.__name__}")
        spinner_kwargs = kwargs.pop('spinner_kwargs', {})
        
        with ThinkingSpinner(text=spinner_text, **spinner_kwargs):
            result = func(*args, **kwargs)
        return result
    
    return wrapper


def run_with_spinner(func: Callable, *args, 
                    text: str = None, 
                    **kwargs) -> Any:
    """
    Run a function with a spinner.
    
    Args:
        func: Function to run
        *args: Arguments to pass to the function
        text: Description for the spinner
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the function
    """
    if text is None:
        text = f"Running {func.__name__}"
        
    with ThinkingSpinner(text=text):
        return func(*args, **kwargs)