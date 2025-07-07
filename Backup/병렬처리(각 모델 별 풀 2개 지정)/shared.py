import threading
from queue import Queue

cached_models = {}
translation_model = None
translation_tokenizer = None

# 모델별 파이프라인 풀 (최대 2개)
pipeline_pools = {}  # {model_name: Queue}
pipeline_pool_locks = {}  # {model_name: threading.Lock}