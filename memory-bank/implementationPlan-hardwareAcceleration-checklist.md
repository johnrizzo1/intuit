# Hardware Acceleration Implementation Checklist

This checklist tracks the implementation progress for the hardware-accelerated audio pipeline migration.

## Phase 1: Foundation & Configuration ⏳

### Hardware Detection (Days 1-2)
- [ ] Create `src/intuit/hardware/__init__.py`
- [ ] Create `src/intuit/hardware/detector.py` with CUDA/MPS/CPU detection
- [ ] Create `src/intuit/hardware/config.py` for hardware configuration
- [ ] Add logging for hardware detection
- [ ] Write unit tests for hardware detection
- [ ] Test on CUDA, MPS, and CPU systems

### Configuration System (Days 1-2)
- [ ] Create `src/intuit/config/__init__.py`
- [ ] Create `src/intuit/config/audio_config.py` with Pydantic models
- [ ] Add environment variable support
- [ ] Create configuration validation
- [ ] Add default configurations
- [ ] Write unit tests for configuration

**Acceptance Criteria:**
- [ ] Hardware detection works on all platforms
- [ ] Configuration loads from environment variables
- [ ] Proper fallback when GPU unavailable
- [ ] All tests pass

---

## Phase 2: Speech-to-Text Migration ⏳

### STT Base Architecture (Days 3-4)
- [ ] Create `src/intuit/audio/__init__.py`
- [ ] Create `src/intuit/audio/stt_base.py` with abstract base class
- [ ] Create `src/intuit/audio/stt_factory.py` for provider selection
- [ ] Write unit tests for base classes

### Whisper Implementation (Days 3-4)
- [ ] Create `src/intuit/audio/stt_whisper.py`
- [ ] Implement GPU acceleration support
- [ ] Add audio preprocessing (resampling, normalization)
- [ ] Implement model warmup
- [ ] Add performance metrics logging
- [ ] Write unit tests for Whisper STT

### Google STT Refactor (Day 4)
- [ ] Create `src/intuit/audio/stt_google.py`
- [ ] Refactor existing Google STT into provider pattern
- [ ] Maintain backward compatibility
- [ ] Write unit tests for Google STT

### Integration (Day 5)
- [ ] Update `src/intuit/ui/voice.py` to use STT factory
- [ ] Update `src/intuit/dictation.py` to use STT factory
- [ ] Add configuration switching
- [ ] Write integration tests
- [ ] Performance benchmarking

**Acceptance Criteria:**
- [ ] Whisper STT works with GPU acceleration
- [ ] Can switch between Whisper and Google via config
- [ ] Voice interface maintains functionality
- [ ] Dictation mode works with new STT
- [ ] 5-10x performance improvement measured
- [ ] All tests pass

---

## Phase 3: Text-to-Speech Migration ⏳

### TTS Base Architecture (Days 1-2)
- [ ] Create `src/intuit/audio/tts_base.py` with abstract base class
- [ ] Create `src/intuit/audio/tts_factory.py` for provider selection
- [ ] Write unit tests for base classes

### Coqui TTS Implementation (Days 1-2)
- [ ] Create `src/intuit/audio/tts_coqui.py`
- [ ] Implement GPU acceleration support
- [ ] Add model warmup
- [ ] Add performance metrics logging
- [ ] Write unit tests for Coqui TTS

### Audio Player (Day 2)
- [ ] Create `src/intuit/audio/audio_player.py`
- [ ] Implement cross-platform playback with sounddevice
- [ ] Replace macOS-specific afplay
- [ ] Add device selection support
- [ ] Write unit tests for audio player

### gTTS Refactor (Day 2)
- [ ] Create `src/intuit/audio/tts_gtts.py`
- [ ] Refactor existing gTTS into provider pattern
- [ ] Maintain backward compatibility
- [ ] Write unit tests for gTTS

### Integration (Day 3)
- [ ] Update `src/intuit/voice.py` to use TTS factory
- [ ] Update `src/intuit/ui/voice.py` to use new audio player
- [ ] Add configuration switching
- [ ] Write integration tests
- [ ] Performance benchmarking

**Acceptance Criteria:**
- [ ] Coqui TTS works with GPU acceleration
- [ ] Audio playback works on macOS, Linux, Windows
- [ ] Can switch between Coqui and gTTS via config
- [ ] Voice quality equal or better than gTTS
- [ ] 10-20x performance improvement measured
- [ ] All tests pass

---

## Phase 4: LLM Migration to Ollama ⏳

### LLM Provider Architecture (Days 4-5)
- [ ] Create `src/intuit/llm/__init__.py`
- [ ] Create `src/intuit/llm/base.py` with provider abstraction
- [ ] Create `src/intuit/llm/factory.py` for provider selection
- [ ] Write unit tests for base classes

### Ollama Implementation (Days 4-5)
- [ ] Create `src/intuit/llm/ollama_provider.py`
- [ ] Implement Ollama integration using langchain-ollama
- [ ] Add model management utilities
- [ ] Add performance metrics logging
- [ ] Write unit tests for Ollama provider

### OpenAI Refactor (Day 5)
- [ ] Create `src/intuit/llm/openai_provider.py`
- [ ] Refactor existing OpenAI into provider pattern
- [ ] Maintain backward compatibility
- [ ] Write unit tests for OpenAI provider

### Integration (Day 5)
- [ ] Update `src/intuit/agent.py` to use LLM factory
- [ ] Add configuration switching
- [ ] Write integration tests
- [ ] Performance benchmarking

**Acceptance Criteria:**
- [ ] Ollama integration works with GPU acceleration
- [ ] Can switch between Ollama and OpenAI via config
- [ ] Model management is user-friendly
- [ ] 2-5x performance improvement measured
- [ ] Response quality comparable to OpenAI
- [ ] All tests pass

---

## Phase 5: Integration & Testing ⏳

### CLI Updates (Days 1-2)
- [ ] Update `src/intuit/main.py` for new configuration
- [ ] Update `src/intuit/cli.py` with hardware info commands
- [ ] Add configuration commands (show, validate, reset)
- [ ] Add model management commands

### Dependency Management (Day 2)
- [x] Update `pyproject.toml` with new dependencies
- [x] Remove redundant `requirements.txt` file
- [ ] Update `devenv.nix` for GPU support
- [x] Update `.env.template` with new options
- [ ] Test dependency installation on all platforms

### End-to-End Testing (Day 3)
- [ ] Write end-to-end integration tests
- [ ] Create performance benchmarking suite
- [ ] Load testing for concurrent requests
- [ ] Test on CUDA, MPS, and CPU systems
- [ ] Verify offline operation

**Acceptance Criteria:**
- [ ] All components work together seamlessly
- [ ] Configuration is intuitive
- [ ] Performance meets targets
- [ ] All integration tests pass
- [ ] Benchmarks show expected improvements

---

## Phase 6: Documentation & Optimization ⏳

### Documentation (Days 4-5)
- [ ] Create `docs/hardware-acceleration.md`
- [ ] Create `docs/configuration.md`
- [ ] Create `docs/performance-tuning.md`
- [ ] Create `docs/troubleshooting.md`
- [ ] Update `README.md` with new features
- [ ] Update `memory-bank/techContext.md`
- [ ] Update `memory-bank/systemPatterns.md`
- [ ] Create migration guide for existing users

### Performance Optimization (Day 5)
- [ ] Profile audio pipeline for bottlenecks
- [ ] Optimize model loading (lazy loading, caching)
- [ ] Implement audio streaming for lower latency
- [ ] Add batch processing support
- [ ] Optimize memory usage
- [ ] Add performance monitoring

**Acceptance Criteria:**
- [ ] Documentation is complete and clear
- [ ] Performance optimized for target hardware
- [ ] Memory usage is reasonable
- [ ] Latency < 500ms end-to-end
- [ ] All optimizations documented

---

## Dependencies to Add

### Python Packages
- [ ] `torch` - PyTorch for GPU acceleration
- [ ] `torchaudio` - Audio processing with GPU support
- [ ] `openai-whisper` - Already in deps, ensure latest version
- [ ] `TTS` (Coqui TTS) - Text-to-speech with GPU support
- [ ] `langchain-ollama` - Ollama integration for LangChain
- [ ] `soundfile` - Already in deps, ensure latest version

### System Dependencies (devenv.nix)
- [ ] CUDA toolkit (for NVIDIA GPUs)
- [ ] Metal support (for Apple Silicon)
- [ ] PortAudio (already included)
- [ ] FFmpeg (for audio format support)

---

## Performance Targets

### Latency Goals
- [ ] STT: < 200ms for 5-second audio clip
- [ ] TTS: < 300ms for 50-word response
- [ ] LLM: < 500ms for typical query
- [ ] End-to-end: < 1000ms total pipeline

### Quality Goals
- [ ] STT accuracy: > 95% for clear speech
- [ ] TTS naturalness: MOS score > 4.0
- [ ] LLM quality: Comparable to OpenAI GPT-4

### Resource Goals
- [ ] GPU memory: < 4GB for base models
- [ ] CPU usage: < 50% during idle
- [ ] Startup time: < 10 seconds

---

## Risk Mitigation

### Technical Risks
- [ ] GPU not available → Fallback to CPU with warning
- [ ] Model download fails → Provide clear error and retry
- [ ] Memory overflow → Implement model unloading
- [ ] Platform incompatibility → Test on all platforms

### User Experience Risks
- [ ] Configuration too complex → Provide sensible defaults
- [ ] Migration breaks existing setup → Maintain backward compatibility
- [ ] Performance worse on CPU → Clear documentation about requirements

---

## Success Metrics

### Must Have
- [ ] All tests passing
- [ ] GPU acceleration working on CUDA and MPS
- [ ] Offline operation confirmed
- [ ] Performance improvement measured and documented
- [ ] No regression in functionality

### Should Have
- [ ] 5-10x STT improvement
- [ ] 10-20x TTS improvement
- [ ] 2-5x LLM improvement
- [ ] < 1s end-to-end latency

### Nice to Have
- [ ] Model auto-download
- [ ] Performance dashboard
- [ ] Multiple model support
- [ ] Voice cloning support

---

## Notes

- Use `tiny` or `base` Whisper models for development/testing
- Test with `llama3.2:3b` for LLM (fastest, good quality)
- Coqui TTS models vary in quality - test multiple
- Always provide CPU fallback for all components
- Log performance metrics for optimization
- Consider memory constraints on consumer hardware

---

## Timeline Summary

| Phase | Duration | Status |
|-------|----------|--------|
| Phase 1: Foundation | 2 days | ⏳ Not Started |
| Phase 2: STT Migration | 3 days | ⏳ Not Started |
| Phase 3: TTS Migration | 3 days | ⏳ Not Started |
| Phase 4: LLM Migration | 2 days | ⏳ Not Started |
| Phase 5: Integration | 3 days | ⏳ Not Started |
| Phase 6: Documentation | 2 days | ⏳ Not Started |
| **Total** | **15 days (3 weeks)** | ⏳ Not Started |

---

## Quick Start Commands (After Implementation)

```bash
# Check hardware capabilities
devenv shell -- intuit hardware-info

# Configure for GPU acceleration
devenv shell -- intuit config set stt.provider whisper
devenv shell -- intuit config set tts.provider coqui
devenv shell -- intuit config set llm.provider ollama

# Download models
devenv shell -- intuit models download whisper-base
devenv shell -- intuit models download coqui-en
devenv shell -- ollama pull llama3.2:3b

# Run with hardware acceleration
devenv shell -- intuit voice --tui

# Benchmark performance
devenv shell -- intuit benchmark