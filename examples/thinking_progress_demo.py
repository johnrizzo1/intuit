#!/usr/bin/env python3
"""
Demo script showing how to use the ThinkingProgress indicator.
This demonstrates different ways to use the progress indicator while the agent is thinking.
"""
import asyncio
import time
import random
from intuit.utils.progress import ThinkingProgress, with_progress, run_with_progress


def simulate_thinking(duration=3):
    """Simulate a thinking process that takes some time."""
    time.sleep(duration)
    return f"Thought about it for {duration} seconds"


@with_progress
def decorated_thinking(duration=2):
    """Function decorated with the progress indicator."""
    time.sleep(duration)
    return f"Decorated thinking for {duration} seconds"


async def async_thinking(duration=2):
    """Async function that simulates thinking."""
    await asyncio.sleep(duration)
    return f"Async thinking for {duration} seconds"


def demo_basic_usage():
    """Demonstrate basic usage with context manager."""
    print("\n=== Basic Usage with Context Manager ===")
    with ThinkingProgress(desc="Basic thinking process"):
        result = simulate_thinking(3)
    print(f"Result: {result}")


def demo_custom_progress():
    """Demonstrate custom progress bar settings."""
    print("\n=== Custom Progress Bar ===")
    with ThinkingProgress(desc="Custom thinking", 
                         total=100, 
                         unit="steps",
                         colour="green"):
        result = simulate_thinking(2)
    print(f"Result: {result}")


def demo_decorator():
    """Demonstrate using the decorator."""
    print("\n=== Using Decorator ===")
    result = decorated_thinking(duration=2, 
                               progress_desc="Thinking with decorator",
                               progress_kwargs={"colour": "blue"})
    print(f"Result: {result}")


def demo_run_with_progress():
    """Demonstrate using the run_with_progress function."""
    print("\n=== Using run_with_progress ===")
    result = run_with_progress(
        simulate_thinking, 
        2.5,  # duration argument
        desc="Running with progress"
    )
    print(f"Result: {result}")


async def demo_async():
    """Demonstrate using with async functions."""
    print("\n=== Using with Async Functions ===")
    with ThinkingProgress(desc="Async thinking process"):
        result = await async_thinking(2)
    print(f"Result: {result}")


def demo_nested_progress():
    """Demonstrate nested progress bars."""
    print("\n=== Nested Progress Bars ===")
    with ThinkingProgress(desc="Outer thinking process", position=0):
        time.sleep(1)
        with ThinkingProgress(desc="Inner thinking process", position=1):
            time.sleep(2)
    print("Nested thinking complete")


if __name__ == "__main__":
    print("=== ThinkingProgress Demo ===")
    print("This demonstrates different ways to show progress while the agent is thinking")
    
    # Run the demos
    demo_basic_usage()
    demo_custom_progress()
    demo_decorator()
    demo_run_with_progress()
    
    # Run the async demo
    asyncio.run(demo_async())
    
    # Run the nested progress demo
    demo_nested_progress()
    
    print("\nAll demos completed!")