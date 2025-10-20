# Performance Logging in Intuit

## Overview

Intuit includes comprehensive performance logging for the audio pipeline, tracking the time taken for each stage of voice interaction. This helps identify bottlenecks and optimize the user experience.

## Pipeline Stages

The voice pipeline consists of four main stages:

1. **STT (Speech-to-Text)**: Converting audio input to text
2. **Agent**: Processing the query and generating a response
3. **TTS (Text-to-Speech)**: Converting the response to audio
4. **Audio Playback**: The duration of the generated audio

## Logged Metrics

### Per-Stage Timing

Each stage logs its duration:

```
[20241020_120000_123456] STT: Complete (2.34s) - Transcribed: 'What is the weather?'
[20241020_120000_123456] AGENT: Complete (1.56s) - Response: 'The weather is...'
[20241020_120000_123456] TTS: Complete (0.89s) - Audio duration: 3.2s - Audio playback finished
```

### Session Summary

At the end of each session, a summary is logged:

```
=== Pipeline Session Ended: 20241020_120000_123456 === [Total: 7.79s | STT: 2.34s | Agent: 1.56s | TTS: 0.89s | Audio: 3.2s]
```

## Timing Breakdown

### STT Duration
- **What it measures**: Time from starting audio capture to completing transcription
- **Includes**: Audio recording (5 seconds), audio processing, and transcription
- **Typical range**: 2-5 seconds (depends on model size and hardware)

### Agent Duration
- **What it measures**: Time from receiving query to generating response
- **Includes**: Query processing, tool execution, LLM inference, and response generation
- **Typical range**: 1-10 seconds (depends on query complexity and model)

### TTS Duration
- **What it measures**: Time from starting synthesis to completing audio generation
- **Includes**: Text processing, audio synthesis, and file writing
- **Typical range**: 0.5-3 seconds (depends on text length and model)

### Audio Duration
- **What it measures**: Length of the generated audio file
- **Includes**: Only the audio content duration (not processing time)
- **Typical range**: 1-10 seconds (depends on response length)

## Viewing Performance Logs

### Default Logging

By default, performance logs are written to `output.log`:

```bash
tail -f output.log
```

### Verbose Mode

Enable verbose logging to see more details:

```bash
intuit voice -v
```

### Filtering Pipeline Logs

To see only pipeline timing information:

```bash
grep "Pipeline Session" output.log
```

To see timing for a specific stage:

```bash
grep "STT: Complete" output.log
grep "AGENT: Complete" output.log
grep "TTS: Complete" output.log
```

## Performance Optimization

### Identifying Bottlenecks

1. **Check session summaries** to see which stage takes the most time
2. **Compare across sessions** to identify patterns
3. **Monitor trends** over time to detect degradation

### Optimization Strategies

#### Slow STT (>5 seconds)
- Use a smaller Whisper model: `STT_MODEL_SIZE=tiny` or `base`
- Enable GPU acceleration if available
- Consider using Google STT for cloud-based processing

#### Slow Agent (>10 seconds)
- Use a smaller LLM model: `llama3.2:1b` instead of `llama3.2:3b`
- Reduce `LLM_MAX_TOKENS` to limit response length
- Enable GPU acceleration for Ollama
- Consider using a faster model like `phi3:mini`

#### Slow TTS (>3 seconds)
- Use gTTS instead of Coqui: `TTS_PROVIDER=gtts`
- Enable GPU acceleration for Coqui if available
- Use a simpler TTS model

### Hardware Acceleration

Performance can be significantly improved with GPU acceleration:

```bash
# For NVIDIA GPUs
STT_DEVICE=cuda
TTS_DEVICE=cuda
TTS_USE_GPU=true

# Verify Ollama is using GPU
ollama ps
```

## Example Performance Analysis

### Good Performance
```
=== Pipeline Session Ended: 20241020_120000 === [Total: 5.5s | STT: 2.1s | Agent: 1.8s | TTS: 0.6s | Audio: 2.5s]
```
- STT: Fast transcription
- Agent: Quick response generation
- TTS: Efficient synthesis
- Total latency: ~3.5s (excluding audio playback)

### Needs Optimization
```
=== Pipeline Session Ended: 20241020_120100 === [Total: 18.2s | STT: 4.5s | Agent: 12.1s | TTS: 1.6s | Audio: 3.8s]
```
- STT: Acceptable
- Agent: **Bottleneck** - Consider smaller model or GPU
- TTS: Acceptable
- Total latency: ~14.4s (too slow)

## Programmatic Access

You can access timing data programmatically:

```python
from intuit.logging_config import PipelineLogger
import logging

logger = logging.getLogger('intuit.pipeline')
pipeline_logger = PipelineLogger(logger)

# Start session
session_id = pipeline_logger.start_session()

# Log stages
pipeline_logger.log_stt_start(session_id)
# ... do STT work ...
pipeline_logger.log_stt_complete("transcribed text", session_id)

# Access timing data
if session_id in pipeline_logger.timings:
    stt_duration = pipeline_logger.timings[session_id].get('stt_duration')
    print(f"STT took {stt_duration:.2f} seconds")

# End session (logs summary)
pipeline_logger.end_session(session_id)
```

## Monitoring Best Practices

1. **Baseline Performance**: Record typical timings for your hardware
2. **Set Thresholds**: Define acceptable ranges for each stage
3. **Regular Monitoring**: Check logs periodically for degradation
4. **Compare Configurations**: Test different models and settings
5. **Document Changes**: Note when you change models or settings

## Troubleshooting

### Missing Timing Data

If timing data is not appearing in logs:

1. Check that logging is enabled: `intuit voice -v`
2. Verify log file location: `output.log` in current directory
3. Ensure pipeline logger is initialized properly

### Inconsistent Timings

If timings vary significantly:

1. **System load**: Check CPU/GPU usage
2. **Model caching**: First run may be slower
3. **Network issues**: For cloud-based services
4. **Disk I/O**: For file operations

### Very Slow Performance

If all stages are slow:

1. Check system resources (CPU, RAM, GPU)
2. Verify Ollama is running: `ollama ps`
3. Check for background processes
4. Consider hardware upgrade or cloud models

## Future Enhancements

Potential improvements to performance logging:

1. **Real-time dashboard**: Web-based performance monitoring
2. **Historical analysis**: Track performance trends over time
3. **Automatic optimization**: Suggest model changes based on performance
4. **Comparative benchmarks**: Compare against baseline performance
5. **Export metrics**: Export to monitoring tools (Prometheus, Grafana)