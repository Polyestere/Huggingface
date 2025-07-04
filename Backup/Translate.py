from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import shared
import torch
import gc
import copy


def init_translation_model():
    if shared.translation_model is None:
        model_name = "facebook/nllb-200-distilled-1.3B"
        shared.translation_tokenizer = AutoTokenizer.from_pretrained(model_name)
        shared.translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("Translation model loaded")


def translate(prompt):
    tokenizer = shared.translation_tokenizer
    model = shared.translation_model


    fresh_tokenizer = copy.deepcopy(tokenizer)
    fresh_tokenizer.src_lang = "kor_Hang"


    if not prompt or not prompt.strip():
        print("[TRANSLATE] Empty prompt, returning empty string")
        return ""


    inputs = fresh_tokenizer(
        prompt.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
        add_special_tokens=True
    )


    target_lang_token = "eng_Latn"
    forced_bos_token_id = fresh_tokenizer.convert_tokens_to_ids(target_lang_token)
    if forced_bos_token_id is None:
        forced_bos_token_id = 2


    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask', None),
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=256,
            min_length=1,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            pad_token_id=fresh_tokenizer.pad_token_id,
            eos_token_id=fresh_tokenizer.eos_token_id,
            use_cache=False,
            repetition_penalty=1.0,
            length_penalty=1.0
        )


    translated_text = fresh_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    ).strip()


    del inputs, outputs, fresh_tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    return translated_text if translated_text else prompt