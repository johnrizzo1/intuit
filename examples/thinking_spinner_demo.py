#!/usr/bin/env python3
"""
Demo script showing how to use the ThinkingSpinner indicator.
This demonstrates different ways to use the spinner indicator while the agent is thinking.
"""
import asyncio
import time
import random
from yaspin.spinners import Spinners
from intuit.utils.spinner import ThinkingSpinner, with_spinner, run_with_spinner


def simulate_thinking(duration=3):
    """Simulate a thinking process that takes some time."""
    time.sleep(duration)
    return f"Thought about it for {duration} seconds"


@with_spinner
def decorated_thinking(duration=2):
    """Function decorated with the spinner indicator."""
    time.sleep(duration)
    return f"Decorated thinking for {duration} seconds"


async def async_thinking(duration=2):
    """Async function that simulates thinking."""
    await asyncio.sleep(duration)
    return f"Async thinking for {duration} seconds"


def demo_basic_usage():
    """Demonstrate basic usage with context manager."""
    print("\n=== Basic Usage with Context Manager ===")
    with ThinkingSpinner(text="Basic thinking process"):
        result = simulate_thinking(3)
    print(f"Result: {result}")


def demo_custom_spinner():
    """Demonstrate custom spinner settings."""
    print("\n=== Custom Spinner ===")
    with ThinkingSpinner(text="Custom thinking", 
                        spinner=Spinners.bouncingBar,
                        color="green") as spinner:
        result = simulate_thinking(2)
    print(f"Result: {result}")


def demo_decorator():
    """Demonstrate using the decorator."""
    print("\n=== Using Decorator ===")
    result = decorated_thinking(duration=2, 
                               spinner_text="Thinking with decorator",
                               spinner_kwargs={"color": "blue"})
    print(f"Result: {result}")


def demo_run_with_spinner():
    """Demonstrate using the run_with_spinner function."""
    print("\n=== Using run_with_spinner ===")
    result = run_with_spinner(
        simulate_thinking, 
        2.5,  # duration argument
        text="Running with spinner"
    )
    print(f"Result: {result}")


async def demo_async():
    """Demonstrate using with async functions."""
    print("\n=== Using with Async Functions ===")
    with ThinkingSpinner(text="Async thinking process"):
        result = await async_thinking(2)
    print(f"Result: {result}")


def demo_writing_messages():
    """Demonstrate writing messages while spinner is active."""
    print("\n=== Writing Messages ===")
    with ThinkingSpinner(text="Processing data") as spinner:
        # Task 1
        time.sleep(1)
        spinner.write("> Step 1 complete")
        
        # Task 2
        time.sleep(1)
        spinner.write("> Step 2 complete")
        
        # Task 3
        time.sleep(1)
        spinner.write("> Step 3 complete")
        
        # Success
        spinner.ok("✓")
    print("All steps completed successfully")


def demo_success_failure():
    """Demonstrate success and failure states."""
    print("\n=== Success and Failure States ===")
    
    # Success case
    with ThinkingSpinner(text="Processing task 1") as spinner:
        time.sleep(1.5)
        spinner.ok("✓")
    print("Task 1 completed successfully")
    
    # Failure case
    with ThinkingSpinner(text="Processing task 2") as spinner:
        time.sleep(1.5)
        spinner.fail("✗")
    print("Task 2 failed")


if __name__ == "__main__":
    print("=== ThinkingSpinner Demo ===")
    print("This demonstrates different ways to show a spinner while the agent is thinking")
    
    # Run the demos
    demo_basic_usage()
    demo_custom_spinner()
    demo_decorator()
    demo_run_with_spinner()
    
    # Run the async demo
    asyncio.run(demo_async())
    
    # Run the writing messages demo
    demo_writing_messages()
    
    # Run the success/failure demo
    demo_success_failure()
    
    print("\nAll demos completed!")