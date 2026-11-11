"""Frozen 7B FP16 free-answer Reader; builder appends a gold-free deduplicated shard."""
from pathlib import Path
import hashlib
import importlib.metadata
import inspect
import json
import random
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


"""Exact attention-head blocking for inference; retains every query/key/value token."""
import torch

def head_blocked_sdpa(sdpa,query,key,value,attn_mask=None,dropout_p=0.0,is_causal=False,*,scale=None,enable_gqa=False):
    if dropout_p != 0.0:
        raise ValueError('Head blocking is inference-only; dropout must be zero')
    qh,kh,vh=query.shape[-3],key.shape[-3],value.shape[-3]
    if kh != vh or (qh != kh and not enable_gqa) or qh % kh:
        raise ValueError('Invalid attention head alignment')
    if attn_mask is not None and attn_mask.ndim not in (2,4):
        raise ValueError('Only native Qwen 2D/4D masks supported')
    if attn_mask is not None and attn_mask.ndim==4 and attn_mask.shape[-3] not in (1,qh):
        raise ValueError('Mask head dimension must broadcast or match query heads')
    ratio=qh//kh
    out=torch.empty((*query.shape[:-1],value.shape[-1]),dtype=query.dtype,device=query.device)
    for h in range(qh):
        mask=attn_mask
        if mask is not None and mask.ndim==4 and mask.shape[-3]==qh:
            mask=mask[...,h:h+1,:,:]
        # Every attention head is independent; a GQA query head maps to floor(h/ratio).
        k=h//ratio
        out[...,h:h+1,:,:]=sdpa(query[...,h:h+1,:,:],key[...,k:k+1,:,:],value[...,k:k+1,:,:],
            attn_mask=mask,dropout_p=dropout_p,is_causal=is_causal,scale=scale,enable_gqa=enable_gqa)
    return out

def install_head_blocked_sdpa(threshold_bytes=256*1024*1024):
    original=torch.nn.functional.scaled_dot_product_attention
    stats=dict(strategy='attention-head blocks of1; native SDPA per block',threshold_bytes=threshold_bytes,
        blocked_calls=0,max_full_score_float32_bytes=0,max_block_score_float32_bytes=0,
        tokens_dropped=0,attention_mask_changed=False)
    def wrapped(query,key,value,attn_mask=None,dropout_p=0.0,is_causal=False,*,scale=None,enable_gqa=False):
        score=query.shape[0]*query.shape[-3]*query.shape[-2]*key.shape[-2]*4
        if query.shape[-2]>1 and score>threshold_bytes:
            stats['blocked_calls']+=1
            stats['max_full_score_float32_bytes']=max(stats['max_full_score_float32_bytes'],score)
            stats['max_block_score_float32_bytes']=max(stats['max_block_score_float32_bytes'],score//query.shape[-3])
            return head_blocked_sdpa(original,query,key,value,attn_mask,dropout_p,is_causal,scale=scale,enable_gqa=enable_gqa)
        return original(query,key,value,attn_mask=attn_mask,dropout_p=dropout_p,is_causal=is_causal,scale=scale,enable_gqa=enable_gqa)
    torch.nn.functional.scaled_dot_product_attention=wrapped
    return stats

def verify_attention_blocking(device='cpu'):
    from torch.nn.attention import sdpa_kernel,SDPBackend
    torch.manual_seed(42);checks=[]
    for dtype in (torch.float32,torch.float16):
        for kind in ('causal_gqa','boolean_gqa','additive_mha','noncausal_gqa','decode_gqa'):
            qlen,klen=(1,11) if kind=='decode_gqa' else (9,9)
            kh=4 if kind=='additive_mha' else 2;qh=4
            q=torch.randn(1,qh,qlen,8,device=device,dtype=dtype)
            k=torch.randn(1,kh,klen,8,device=device,dtype=dtype);v=torch.randn_like(k)
            mask=None
            if kind=='boolean_gqa':mask=torch.ones(1,1,qlen,klen,device=device,dtype=torch.bool).tril()
            if kind=='additive_mha':
                mask=torch.zeros(1,qh,qlen,klen,device=device,dtype=dtype);mask[:,:,:,4:]=-10000
            causal=kind=='causal_gqa';gqa=kh!=qh
            # Match the math-kernel path implicated in the T4 GQA prefill OOM.
            with sdpa_kernel(SDPBackend.MATH):
                a=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask,is_causal=causal,enable_gqa=gqa,scale=.25)
                b=head_blocked_sdpa(torch.nn.functional.scaled_dot_product_attention,q,k,v,attn_mask=mask,is_causal=causal,enable_gqa=gqa,scale=.25)
            tol=1e-6 if dtype==torch.float32 else 1e-3
            torch.testing.assert_close(a,b,rtol=tol,atol=tol)
            checks.append(dict(kind=kind,dtype=str(dtype),max_absolute_error=float((a-b).abs().max()),bitwise_equal=bool(torch.equal(a,b)),tolerance=tol))
    return dict(device=device,seed=42,cases=checks,pass_all=True,full_model_bitwise_equivalence='NOT_CLAIMED; same mathematical attention, independent head scheduling; CPU tests are not T4 measurements')

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def run(payload, model_dir, out):
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model = None
    manifest = dict(stage='starting', shard_id=payload['shard_id'], seed=42,
                    reader='Qwen2.5-7B-Instruct', dtype='float16',
                    model_source=payload['deployment']['model_source'],
                    backend='transformers balanced device_map', payload_sha256=digest(payload),
                    max_prompt_tokens=24576, max_new_tokens=256, do_sample=False, batch_size=1,
                    expected_outputs=len(payload['requests']), outputs=0, skipped_requests=0,
                    official_GPT52_scores=False, api_judge_called=False,
                    packages={x:importlib.metadata.version(x) for x in ['torch','transformers','accelerate','safetensors']})
    try:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        if torch.cuda.device_count() != 2 or any('T4' not in torch.cuda.get_device_name(i) for i in range(2)):
            raise RuntimeError('Requires two independent T4 devices')
        expected = payload['deployment']
        if manifest['packages'] != expected['packages']:
            raise RuntimeError('Software differs from frozen Reader; record and resolve before inference')
        config = model_dir / 'config.json'
        if hashlib.sha256(config.read_bytes()).hexdigest() != expected['config_sha256']:
            raise RuntimeError('Reader model configuration differs from the frozen configuration')
        manifest['config_sha256'] = expected['config_sha256']
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        contexts = {c['context_key']:c['text'] for c in payload['contexts']}
        prepared = []
        seen_messages, seen_ids = set(), set()
        for item in payload['requests']:
            messages = [dict(role='system', content=payload['system_prompts'][item['prompt_id']]),
                        dict(role='user', content=f"<context>\n{contexts[item['context_key']]}\n</context>\n\nQuestion: {item['question']}\nAnswer:")]
            signature = digest(messages)
            request_key = digest(dict(messages=messages,generation=payload['generation_config'],max_new_tokens=item['max_new_tokens']))
            if request_key != item['request_id']: raise RuntimeError('Request generation/config hash mismatch')
            if signature != item['messages_sha256'] or item['request_id'] in seen_ids or (signature,item['max_new_tokens']) in seen_messages:
                raise RuntimeError('Input corruption or duplicate GPU input')
            seen_ids.add(item['request_id'])
            seen_messages.add((signature,item['max_new_tokens']))
            if item['max_new_tokens'] not in [128,256]: raise RuntimeError('Invalid factorial generation cap')
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            tokens = tokenizer(text, add_special_tokens=False)['input_ids']
            if len(tokens) > manifest['max_prompt_tokens']:
                raise RuntimeError('Prompt exceeds frozen limit; no truncation or per-arm skipping allowed')
            prepared.append(tokens)
        manifest['prompt_tokens_range'] = [min(map(len,prepared)), max(map(len,prepared))]
        manifest['stage'] = 'loading_model'
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        # Bounded numerical probe; no model requests and no original answers are recomputed.
        math_probe=verify_attention_blocking('cuda:0')
        (out/'attention_gpu_tests.json').write_text(json.dumps(math_probe,indent=2),encoding='utf-8')
        workspace_stats=install_head_blocked_sdpa()
        manifest['attention_workspace']=workspace_stats
        manifest['reused_P0_128_outputs']=0
        manifest['primary_cap']=128
        manifest['diagnostic_cap']=256
        manifest['generated_token_ids_saved']=True
        manifest['final_protocol_sha256']=payload['protocol_sha256']
        manifest['numerical_equivalence']='Independent-head partition; native SDPA, full keys and unchanged mask. Tensor tests pass; full-model bitwise identity across scheduling NOT_CLAIMED.'
        model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False,
                    dtype=torch.float16, device_map='balanced', max_memory={0:'13GiB',1:'13GiB'},
                    attn_implementation='sdpa').eval()
        if next(model.parameters()).dtype != torch.float16:
            raise RuntimeError('Reader must be FP16')
        manifest['device_map'] = {k:str(v) for k,v in model.hf_device_map.items()}
        if manifest['device_map'] != expected['device_map']:
            raise RuntimeError('Layer placement differs from frozen Reader')
        manifest['model_load_seconds'] = time.perf_counter()-started
        forward = inspect.signature(model.forward).parameters
        extra = {'logits_to_keep':1} if 'logits_to_keep' in forward else ({'num_logits_to_keep':1} if 'num_logits_to_keep' in forward else {})
        manifest['stage'] = 'inference'
        with (out/'predictions.jsonl').open('w', encoding='utf-8') as output:
            for index,(item,tokens) in enumerate(zip(payload['requests'], prepared)):
                if time.perf_counter()-started >= 10*3600:
                    manifest['checkpointed_for_time_margin']=True
                    break
                manifest['active_request_id'] = item['request_id']
                (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
                inputs = torch.tensor([tokens], dtype=torch.long, device=model.device)
                begin = time.perf_counter()
                with torch.inference_mode():
                    ids = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                         max_new_tokens=item['max_new_tokens'], do_sample=False, use_cache=True,
                                         pad_token_id=tokenizer.eos_token_id, **extra)
                generated = ids[0,len(tokens):]
                generated_token_ids = generated.tolist()
                row = dict(item, raw_response=tokenizer.decode(generated_token_ids, skip_special_tokens=True),
                           generated_token_ids=generated_token_ids,
                           primary128_token_ids=generated_token_ids[:128],
                           primary128_response=tokenizer.decode(generated_token_ids[:128], skip_special_tokens=True),
                           primary128_hit_cap=len(generated)>=128,
                           actual_eos_last=bool(len(generated_token_ids) and generated_token_ids[-1] in (model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id,list) else [model.generation_config.eos_token_id])),
                           prompt_tokens=len(tokens), generated_tokens=len(generated),
                           hit_generation_cap=len(generated)==item['max_new_tokens'],
                           seconds=time.perf_counter()-begin)
                output.write(json.dumps(row, ensure_ascii=False)+'\n')
                output.flush()
                manifest['outputs'] += 1
                del inputs, ids, generated
                torch.cuda.empty_cache()
                if (index+1)%10 == 0:
                    print(f"{index+1}/{len(prepared)}", flush=True)
        manifest.update(stage='completed' if manifest['outputs']==manifest['expected_outputs'] else 'checkpointed_time_margin', peak_allocated_bytes=[torch.cuda.max_memory_allocated(i) for i in range(2)])
    except Exception as error:
        manifest.update(stage='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        manifest['seconds'] = time.perf_counter()-started
        (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        if model is not None:
            del model
        torch.cuda.empty_cache()
        print(json.dumps(manifest, ensure_ascii=False), flush=True)





def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the frozen FP16 reader on a prepared request bundle")
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be empty to preserve earlier predictions")
    run(json.loads(args.payload.read_text(encoding="utf-8")), args.model_dir, args.output)


if __name__ == "__main__":
    main()
