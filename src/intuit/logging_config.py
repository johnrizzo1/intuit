"""
Logging configuration for Intuit.

This module provides centralized logging configuration with support for:
- File-based logging (output.log)
- Debug mode for pipeline tracking
- Stage-based logging for STT -> Agent -> TTS pipeline
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class PipelineLogger:
    """Logger for tracking the voice processing pipeline stages."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.current_session_id: Optional[str] = None
    
    def start_session(self) -> str:
        """Start a new pipeline session and return session ID."""
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.logger.info(
            f"=== Pipeline Session Started: {self.current_session_id} ==="
        )
        return self.current_session_id
    
    def log_stt_start(self, session_id: Optional[str] = None):
        """Log the start of Speech-to-Text processing."""
        sid = session_id or self.current_session_id
        self.logger.info(f"[{sid}] STT: Starting speech recognition...")
    
    def log_stt_complete(self, text: str, session_id: Optional[str] = None):
        """Log successful STT completion with transcribed text."""
        sid = session_id or self.current_session_id
        self.logger.info(f"[{sid}] STT: Complete - Transcribed: '{text}'")
    
    def log_stt_error(self, error: str, session_id: Optional[str] = None):
        """Log STT error."""
        sid = session_id or self.current_session_id
        self.logger.error(f"[{sid}] STT: Error - {error}")
    
    def log_agent_start(self, query: str, session_id: Optional[str] = None):
        """Log the start of agent processing."""
        sid = session_id or self.current_session_id
        self.logger.info(f"[{sid}] AGENT: Processing query: '{query}'")
    
    def log_agent_complete(
        self, response: str, session_id: Optional[str] = None
    ):
        """Log successful agent completion with response."""
        sid = session_id or self.current_session_id
        response_preview = response[:100] if len(response) > 100 else response
        self.logger.info(
            f"[{sid}] AGENT: Complete - Response: '{response_preview}...'"
        )
    
    def log_agent_error(self, error: str, session_id: Optional[str] = None):
        """Log agent error."""
        sid = session_id or self.current_session_id
        self.logger.error(f"[{sid}] AGENT: Error - {error}")
    
    def log_tts_start(self, text: str, session_id: Optional[str] = None):
        """Log the start of Text-to-Speech processing."""
        sid = session_id or self.current_session_id
        text_preview = text[:50] if len(text) > 50 else text
        self.logger.info(
            f"[{sid}] TTS: Starting speech synthesis for: '{text_preview}...'"
        )
    
    def log_tts_complete(self, session_id: Optional[str] = None):
        """Log successful TTS completion."""
        sid = session_id or self.current_session_id
        self.logger.info(
            f"[{sid}] TTS: Complete - Audio playback finished"
        )
    
    def log_tts_error(self, error: str, session_id: Optional[str] = None):
        """Log TTS error."""
        sid = session_id or self.current_session_id
        self.logger.error(f"[{sid}] TTS: Error - {error}")
    
    def end_session(self, session_id: Optional[str] = None):
        """End the current pipeline session."""
        sid = session_id or self.current_session_id
        self.logger.info(f"=== Pipeline Session Ended: {sid} ===\n")
        self.current_session_id = None


def setup_logging(
    log_file: str = "output.log",
    debug_mode: bool = False,
    console_output: bool = False
) -> PipelineLogger:
    """
    Configure logging for Intuit.
    
    Args:
        log_file: Path to log file (default: output.log)
        debug_mode: Enable debug level logging (default: False)
        console_output: Also output to console (default: False)
    
    Returns:
        PipelineLogger instance for pipeline tracking
    """
    # Determine log level
    log_level = logging.DEBUG if debug_mode else logging.INFO
    
    # Create log directory if needed
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger FIRST
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove ALL existing handlers from root logger
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)
    
    # Clear all filters from root logger
    root_logger.filters.clear()
    
    # Remove handlers from ALL existing loggers
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        logger.filters.clear()
    
    # Disable the lastResort handler
    logging.lastResort = None
    
    # Redirect stdout and stderr to log file if console output is disabled
    if not console_output:
        log_file_handle = open(log_file, 'a', encoding='utf-8')
        sys.stdout = log_file_handle
        sys.stderr = log_file_handle
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler (always enabled)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler (optional)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
    
    # Configure specific loggers - clear their handlers too
    loggers_to_configure = [
        'intuit',
        'intuit.agent',
        'intuit.dictation',
        'intuit.ui.voice',
        'intuit.ui.voice_tui',
        'intuit.voice',
    ]
    
    for logger_name in loggers_to_configure:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        logger.propagate = True
        # Remove any existing handlers from these loggers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        # Clear filters
        logger.filters.clear()
    
    # Suppress noisy third-party loggers unless in debug mode
    if not debug_mode:
        noisy_loggers = [
            'chromadb',
            'urllib3',
            'httpx',
            'httpcore',
            'asyncio',
        ]
        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Create pipeline logger
    intuit_logger = logging.getLogger('intuit.pipeline')
    intuit_logger.setLevel(log_level)
    pipeline_logger = PipelineLogger(intuit_logger)
    
    # Log startup message
    root_logger.info("=" * 80)
    root_logger.info(f"Intuit Logging Initialized - Debug Mode: {debug_mode}")
    root_logger.info(f"Log File: {log_path.absolute()}")
    root_logger.info("=" * 80)
    
    return pipeline_logger


def get_pipeline_logger() -> PipelineLogger:
    """Get or create the pipeline logger instance."""
    logger = logging.getLogger('intuit.pipeline')
    return PipelineLogger(logger)