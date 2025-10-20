"""Unit tests for hardware detector."""
from unittest.mock import Mock, patch
from intuit.hardware.detector import (
    HardwareDetector,
    HardwareCapabilities,
    AcceleratorType,
)


class TestHardwareDetector:
    """Test hardware detection functionality."""
    
    def test_accelerator_type_enum(self):
        """Test AcceleratorType enum values."""
        assert AcceleratorType.CUDA.value == "cuda"
        assert AcceleratorType.MPS.value == "mps"
        assert AcceleratorType.CPU.value == "cpu"
    
    def test_hardware_capabilities_dataclass(self):
        """Test HardwareCapabilities dataclass."""
        caps = HardwareCapabilities(
            accelerator=AcceleratorType.CUDA,
            device_name="NVIDIA GeForce RTX 3080",
            compute_capability="8.6",
            memory_gb=10.0,
            supports_fp16=True,
            supports_int8=True,
        )
        
        assert caps.accelerator == AcceleratorType.CUDA
        assert caps.device_name == "NVIDIA GeForce RTX 3080"
        assert caps.compute_capability == "8.6"
        assert caps.memory_gb == 10.0
        assert caps.supports_fp16 is True
        assert caps.supports_int8 is True
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    def test_detect_cuda(self, mock_get_props, mock_cuda_available):
        """Test CUDA detection."""
        # Mock CUDA availability
        mock_cuda_available.return_value = True
        
        # Mock device properties
        mock_device = Mock()
        mock_device.name = "NVIDIA GeForce RTX 3080"
        mock_device.major = 8
        mock_device.minor = 6
        mock_device.total_memory = 10 * 1024**3  # 10 GB
        mock_get_props.return_value = mock_device
        
        caps = HardwareDetector.detect()
        
        assert caps.accelerator == AcceleratorType.CUDA
        assert caps.device_name == "NVIDIA GeForce RTX 3080"
        assert caps.compute_capability == "8.6"
        assert caps.memory_gb == 10.0
        assert caps.supports_fp16 is True
        assert caps.supports_int8 is True
    
    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    @patch('platform.processor')
    def test_detect_mps(
        self,
        mock_processor,
        mock_mps_available,
        mock_cuda_available
    ):
        """Test MPS (Apple Metal) detection."""
        # Mock no CUDA but MPS available
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = True
        mock_processor.return_value = "Apple M1"
        
        caps = HardwareDetector.detect()
        
        assert caps.accelerator == AcceleratorType.MPS
        assert caps.device_name == "Apple M1"
        assert caps.compute_capability is None
        assert caps.memory_gb == 0.0  # MPS shares system memory
        assert caps.supports_fp16 is True
        assert caps.supports_int8 is False
    
    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    @patch('platform.processor')
    def test_detect_cpu_fallback(
        self,
        mock_processor,
        mock_mps_available,
        mock_cuda_available
    ):
        """Test CPU fallback when no GPU available."""
        # Mock no GPU available
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False
        mock_processor.return_value = "Intel Core i7"
        
        caps = HardwareDetector.detect()
        
        assert caps.accelerator == AcceleratorType.CPU
        assert caps.device_name == "Intel Core i7"
        assert caps.compute_capability is None
        assert caps.memory_gb == 0.0
        assert caps.supports_fp16 is False
        assert caps.supports_int8 is False
    
    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_get_device_string(self, mock_mps_available, mock_cuda_available):
        """Test get_device_string method."""
        # Test CPU
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False
        assert HardwareDetector.get_device_string() == "cpu"
    
    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_is_gpu_available(self, mock_mps_available, mock_cuda_available):
        """Test is_gpu_available method."""
        # Test with CUDA
        mock_cuda_available.return_value = True
        mock_mps_available.return_value = False
        assert HardwareDetector.is_gpu_available() is True
        
        # Test with MPS
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = True
        assert HardwareDetector.is_gpu_available() is True
        
        # Test with no GPU
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False
        assert HardwareDetector.is_gpu_available() is False
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    def test_get_recommended_compute_type_fp16(
        self,
        mock_get_props,
        mock_cuda_available
    ):
        """Test recommended compute type with FP16 support."""
        mock_cuda_available.return_value = True
        mock_device = Mock()
        mock_device.name = "NVIDIA GeForce RTX 3080"
        mock_device.major = 8
        mock_device.minor = 6
        mock_device.total_memory = 10 * 1024**3
        mock_get_props.return_value = mock_device
        
        compute_type = HardwareDetector.get_recommended_compute_type()
        assert compute_type == "float16"
    
    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_get_recommended_compute_type_fp32(
        self,
        mock_mps_available,
        mock_cuda_available
    ):
        """Test recommended compute type without FP16 support."""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False
        
        compute_type = HardwareDetector.get_recommended_compute_type()
        assert compute_type == "float32"
    
    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    @patch('platform.processor')
    @patch('builtins.print')
    def test_print_capabilities(
        self,
        mock_print,
        mock_processor,
        mock_mps_available,
        mock_cuda_available
    ):
        """Test print_capabilities method."""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False
        mock_processor.return_value = "Intel Core i7"
        
        HardwareDetector.print_capabilities()
        
        # Verify print was called
        assert mock_print.call_count > 0
        
        # Check that key information was printed
        calls = [str(call) for call in mock_print.call_args_list]
        output = " ".join(calls)
        assert "Hardware Capabilities" in output
        assert "CPU" in output