# Intuit Examples

This directory contains example scripts demonstrating various features of the Intuit project.

## Example Scripts

### 1. ThinkingSpinner Demo

The `thinking_spinner_demo.py` script demonstrates the basic usage of the spinner indicator using yaspin.

### 2. Memory System with Spinner Indicators

The `memory_spinner_demo.py` script demonstrates the integration of spinner indicators with the memory system.

## ThinkingSpinner Demo

The `thinking_spinner_demo.py` script demonstrates how to use the `ThinkingSpinner` indicator to show a spinner while the agent is thinking or processing.

### Running the Demo

```bash
python examples/thinking_spinner_demo.py
```

### Features Demonstrated

1. **Basic Usage with Context Manager**
   ```python
   with ThinkingSpinner(text="Basic thinking process"):
       result = simulate_thinking(3)
   ```

2. **Custom Spinner Settings**
   ```python
   with ThinkingSpinner(text="Custom thinking",
                       spinner=Spinners.bouncingBar,
                       color="green"):
       result = simulate_thinking(2)
   ```

3. **Using the Decorator**
   ```python
   @with_spinner
   def decorated_thinking(duration=2):
       time.sleep(duration)
       return f"Decorated thinking for {duration} seconds"
   ```

4. **Using run_with_spinner Function**
   ```python
   result = run_with_spinner(
       simulate_thinking,
       2.5,  # duration argument
       text="Running with spinner"
   )
   ```

5. **Using with Async Functions**
   ```python
   async def demo_async():
       with ThinkingSpinner(text="Async thinking process"):
           result = await async_thinking(2)
   ```

6. **Writing Messages**
   ```python
   with ThinkingSpinner(text="Processing data") as spinner:
       # Do work
       spinner.write("> Step 1 complete")
       # Do more work
   ```

7. **Success and Failure States**
   ```python
   with ThinkingSpinner(text="Processing task") as spinner:
       # Do work
       if success:
           spinner.ok("✓")
       else:
           spinner.fail("✗")
   ```

## Integration with Memory System

The spinner indicator is integrated with the memory system to provide visual feedback during long-running operations:

1. **Processing Conversations**
   - Shows a spinner while the agent processes conversation data
   - Integrated in `IntuitMemoryManager.process_conversation()`

2. **Searching Memories**
   - Shows a spinner while searching through memories
   - Integrated in `ChromaMemoryStore.search_memories()`

3. **Summarizing Conversations**
   - Shows a spinner while generating conversation summaries
   - Integrated in `IntuitMemoryManager.summarize_conversation()`

### Running the Memory Spinner Demo

```bash
python examples/memory_spinner_demo.py
```

This demo:
1. Creates a temporary memory store
2. Adds sample memories
3. Searches for memories with a spinner indicator
4. Processes a conversation with a spinner indicator
5. Summarizes a conversation with a spinner indicator
6. Cleans up temporary files

## Customizing the Spinner Indicator

The `ThinkingSpinner` class accepts all the parameters that `yaspin` accepts, including:

- `text`: Text to display next to the spinner
- `spinner`: Spinner type to use (from yaspin.spinners.Spinners)
- `color`: Color of the spinner (e.g., "green", "blue")
- `on_color`: Background color (e.g., "on_red", "on_blue")
- `attrs`: Text attributes (e.g., ["bold", "blink"])

For more options, see the [yaspin documentation](https://github.com/pavdmyt/yaspin).