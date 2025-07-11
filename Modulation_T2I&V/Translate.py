from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import shared
import torch
import gc

def init_translation_model():
    """번역 모델 초기화"""
    if shared.translation_model is None:
        print("[TRANSLATE] 번역 모델 로딩 시작...")
        
        model_name = "facebook/nllb-200-distilled-1.3B"
        
        try:
            shared.translation_tokenizer = AutoTokenizer.from_pretrained(model_name)
            shared.translation_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # 메모리 효율성을 위해 float16 사용
                low_cpu_mem_usage=True      # CPU 메모리 사용량 최적화
            ).to('cpu')
            
            print("[TRANSLATE] 번역 모델 로딩 완료 (CPU 모드)")
            
        except Exception as e:
            print(f"[TRANSLATE] 번역 모델 로딩 실패: {e}")
            raise e

def translate(prompt: str) -> str:
    """메모리 효율적인 번역 함수"""
    if not prompt or not prompt.strip():
        print("[TRANSLATE] 빈 프롬프트, 빈 문자열 반환")
        return ""
    
    if shared.translation_model is None or shared.translation_tokenizer is None:
        print("[TRANSLATE] 번역 모델이 로드되지 않음")
        return prompt
    
    tokenizer = shared.translation_tokenizer
    model = shared.translation_model
    
    # 원본 언어 설정 백업
    orig_src_lang = getattr(tokenizer, 'src_lang', None)
    
    try:
        tokenizer.src_lang = "kor_Hang"
        
        # 입력 토큰화 (메모리 효율적인 설정)
        inputs = tokenizer(
            prompt.strip(),
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
            add_special_tokens=True
        )
        
        # 타겟 언어 토큰 설정
        target_lang_token = "eng_Latn"
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang_token)
        if forced_bos_token_id is None:
            forced_bos_token_id = 2
        
        # CPU에서 추론 (메모리 효율적인 설정)
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'].to('cpu'),
                attention_mask=inputs.get('attention_mask', None).to('cpu') if 'attention_mask' in inputs else None,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=256,
                min_length=1,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,  # 메모리 절약을 위해 캐시 비활성화
                repetition_penalty=1.0,
                length_penalty=1.0
            )
        
        # 결과 디코딩
        translated_text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()
        
        # 메모리 정리
        del inputs, outputs
        gc.collect()
        
        result = translated_text if translated_text else prompt
        print(f"[TRANSLATE] '{prompt}' -> '{result}'")
        
        return result
        
    except Exception as e:
        print(f"[TRANSLATE] 번역 중 오류: {e}")
        return prompt
        
    finally:
        # 원본 언어 설정 복원
        if orig_src_lang is not None:
            tokenizer.src_lang = orig_src_lang
