import threading
import torch
import gc
import time
import psutil
from queue import Queue
from typing import Dict, Optional

# 기존 변수들
cached_models = {}
translation_model = None
translation_tokenizer = None

# 모델별 파이프라인 풀
pipeline_pools = {}  # {model_name: Queue}
pipeline_pool_locks = {}  # {model_name: threading.Lock}

# 새로운 메모리 관리 변수들
GPU_MEMORY_LOCK = threading.Lock()
MEMORY_THRESHOLD = 0.85  # GPU 메모리 사용률 임계값 (85%)
CLEANUP_INTERVAL = 300   # 메모리 정리 간격 (5분)
LOW_MEMORY_THRESHOLD = 0.9  # 낮은 해상도 모드 전환 임계값

class MemoryMonitor:
    """GPU 메모리 모니터링 및 관리 클래스"""
    
    def __init__(self):
        self.last_cleanup = 0
        self.cleanup_count = 0
        
    def get_gpu_memory_usage(self) -> float:
        """GPU 메모리 사용률 반환 (0.0 ~ 1.0)"""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            
            return max(allocated, reserved) / total
        except Exception as e:
            print(f"[MEMORY] GPU 메모리 정보 조회 실패: {e}")
            return 0.0
    
    def get_gpu_memory_info(self) -> dict:
        """상세한 GPU 메모리 정보 반환"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        try:
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            
            return {
                "allocated_gb": allocated / (1024**3),
                "reserved_gb": reserved / (1024**3),
                "total_gb": total / (1024**3),
                "usage_percent": (max(allocated, reserved) / total) * 100,
                "free_gb": (total - max(allocated, reserved)) / (1024**3)
            }
        except Exception as e:
            return {"error": f"메모리 정보 조회 실패: {e}"}
    
    def should_cleanup(self) -> bool:
        """메모리 정리가 필요한지 확인"""
        current_time = time.time()
        
        # 임계값 초과 또는 정기 정리 시간 도달
        return (self.get_gpu_memory_usage() > MEMORY_THRESHOLD or 
                current_time - self.last_cleanup > CLEANUP_INTERVAL)
    
    def cleanup_memory(self, force: bool = False):
        """메모리 정리 수행"""
        if not force and not self.should_cleanup():
            return False
        
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            gc.collect()
            self.last_cleanup = time.time()
            self.cleanup_count += 1
            
            print(f"[MEMORY] 메모리 정리 완료 (#{self.cleanup_count}) - "
                  f"사용률: {self.get_gpu_memory_usage():.2%}")
            return True
            
        except Exception as e:
            print(f"[MEMORY] 메모리 정리 중 오류: {e}")
            return False
    
    def is_low_memory(self) -> bool:
        """낮은 메모리 상태인지 확인"""
        return self.get_gpu_memory_usage() > LOW_MEMORY_THRESHOLD

# 전역 메모리 모니터 인스턴스
memory_monitor = MemoryMonitor()

def get_system_info():
    """시스템 정보 조회"""
    return {
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent
    }
