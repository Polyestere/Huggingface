from model_manager.factory import get_model_instance
import shared
import torch
import gc
from queue import Queue, Empty
import threading
import contextlib
from typing import Optional, Generator
import time

class PipelineManager:
    """파이프라인 관리 클래스"""
    
    def __init__(self):
        self.acquisition_timeout = 60.0
        self.max_retries = 3
        self.current_model_name = None  # 현재 로드된 모델 추적

    def model_load(self, model_name: str):
        """동일한 모델 요청 시 재사용, 다른 모델 요청 시 교체"""
        with shared.GPU_MEMORY_LOCK:
            # 메모리 정리 확인
            if shared.memory_monitor.should_cleanup():
                shared.memory_monitor.cleanup_memory()

            # 동일한 모델이 이미 로드되어 있으면 재사용
            if (self.current_model_name == model_name and 
                model_name in shared.cached_models):
                print(f"[LOADER] {model_name} 이미 로드되어 있음 - 재사용")
                return shared.cached_models[model_name]

            # 다른 모델이 로드되어 있으면 해제
            if (self.current_model_name and 
                self.current_model_name != model_name and 
                self.current_model_name in shared.cached_models):
                print(f"[LOADER] {self.current_model_name} 모델 해제 후 {model_name} 로드")
                
                # 기존 모델 해제
                del shared.cached_models[self.current_model_name]
                
                # 관련 파이프라인 풀도 함께 해제
                if self.current_model_name in shared.pipeline_pools:
                    del shared.pipeline_pools[self.current_model_name]
                if self.current_model_name in shared.pipeline_pool_locks:
                    del shared.pipeline_pool_locks[self.current_model_name]
                
                # 강제 메모리 정리
                shared.memory_monitor.cleanup_memory(force=True)

            print(f"[LOADER] {model_name} 로딩 시작")

            try:
                model = get_model_instance(model_name)
                pipe = model.load()
                shared.cached_models[model_name] = pipe
                self.current_model_name = model_name  # 현재 모델 업데이트

                final_memory = shared.memory_monitor.get_gpu_memory_usage()
                print(f"[LOADER] {model_name} 로딩 완료 - 메모리 사용률: {final_memory:.2%}")
                return pipe

            except Exception as e:
                print(f"[LOADER] {model_name} 로딩 실패: {e}")
                shared.memory_monitor.cleanup_memory(force=True)
                raise e

    
    def model_load_multi(self, model_name: str, num_instances: int = 1):
        """다중 인스턴스 파이프라인 풀 생성 (동일 모델 재사용)"""
        # 동일한 모델의 풀이 이미 존재하면 재사용
        if (model_name == self.current_model_name and 
            model_name in shared.pipeline_pools):
            print(f"[POOL] {model_name} 기존 파이프라인 풀 재사용")
            return

        # 다른 모델의 풀이 있으면 해제
        if (self.current_model_name and 
            self.current_model_name != model_name and 
            self.current_model_name in shared.pipeline_pools):
            print(f"[POOL] {self.current_model_name} 기존 파이프라인 풀 해제")
            del shared.pipeline_pools[self.current_model_name]
            del shared.pipeline_pool_locks[self.current_model_name]
        
        # 새로 로드 (model_load에서 이미 재사용 로직 처리됨)
        base_pipe = self.model_load(model_name)
        
        pool = Queue(maxsize=num_instances)
        pool.put(base_pipe)
        shared.pipeline_pools[model_name] = pool
        shared.pipeline_pool_locks[model_name] = threading.Lock()
        
        print(f"[POOL] {model_name} 파이프라인 풀 생성 완료 (크기: {num_instances})")

    
    @contextlib.contextmanager
    def acquire_pipeline_context(self, model_name: str) -> Generator:
        """컨텍스트 매니저를 사용한 안전한 파이프라인 관리"""
        pipe = None
        start_time = time.time()
        
        try:
            pipe = self.acquire_pipeline(model_name)
            acquisition_time = time.time() - start_time
            print(f"[PIPELINE] {model_name} 획득 완료 (소요시간: {acquisition_time:.2f}초)")
            yield pipe
            
        except Exception as e:
            print(f"[PIPELINE] {model_name} 사용 중 오류: {e}")
            raise e
            
        finally:
            if pipe is not None:
                self.release_pipeline(model_name, pipe)
                release_time = time.time() - start_time
                print(f"[PIPELINE] {model_name} 해제 완료 (총 사용시간: {release_time:.2f}초)")
    
    def acquire_pipeline(self, model_name: str, timeout: float = None) -> Optional[object]:
        """파이프라인 획득 (타임아웃 및 재시도 로직 포함)"""
        if timeout is None:
            timeout = self.acquisition_timeout
            
        if model_name not in shared.pipeline_pools:
            self.model_load_multi(model_name)
        
        pool = shared.pipeline_pools[model_name]
        
        for attempt in range(self.max_retries):
            try:
                pipe = pool.get(timeout=timeout)
                return pipe
                
            except Empty:
                if attempt < self.max_retries - 1:
                    print(f"[PIPELINE] {model_name} 획득 재시도 ({attempt + 1}/{self.max_retries})")
                    time.sleep(1)
                else:
                    raise TimeoutError(f"파이프라인 획득 타임아웃: {model_name} (대기시간: {timeout}초)")
    
    def release_pipeline(self, model_name: str, pipe):
        """파이프라인 해제"""
        if pipe is None:
            return
            
        pool = shared.pipeline_pools.get(model_name)
        if pool:
            try:
                pool.put_nowait(pipe)
            except:
                print(f"[PIPELINE] {model_name} 파이프라인 해제 실패 - 풀이 가득참")
    
    def clear_model_cache(self):
        """모델 캐시 정리"""
        with shared.GPU_MEMORY_LOCK:
            shared.cached_models.clear()
            shared.pipeline_pools.clear()
            shared.pipeline_pool_locks.clear()
            self.current_model_name = None  # 현재 모델 추적 초기화
            shared.memory_monitor.cleanup_memory(force=True)
            print("[CACHE] 모델 캐시 정리 완료")

# 전역 인스턴스
pipeline_manager = PipelineManager()

# 기존 함수들을 래퍼로 유지 (하위 호환성)
def model_load(model_name: str):
    return pipeline_manager.model_load(model_name)

def model_load_multi(model_name: str, num_instances: int = 1):
    return pipeline_manager.model_load_multi(model_name, num_instances)

def acquire_pipeline(model_name: str, timeout: float = 60.0):
    return pipeline_manager.acquire_pipeline(model_name, timeout)

def release_pipeline(model_name: str, pipe):
    return pipeline_manager.release_pipeline(model_name, pipe)

def clear_model_cache():
    return pipeline_manager.clear_model_cache()

def get_model_info():
    """모델 정보 조회"""
    info = {}
    for model_name, pipe in shared.cached_models.items():
        info[model_name] = {
            "loaded": True,
            "device": str(pipe.device) if hasattr(pipe, 'device') else "unknown",
            "dtype": str(pipe.dtype) if hasattr(pipe, 'dtype') else "unknown",
            "pool_size": shared.pipeline_pools[model_name].qsize() if model_name in shared.pipeline_pools else 0
        }
    return info
