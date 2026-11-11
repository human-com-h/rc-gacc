"""Local frozen DeBERTa NLI diagnostics, not the KEEP official LLM judge."""
from pathlib import Path
import json
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

REVISION='6bc2a55c7c0f7e2bc68de60bb248e523e2612abb'

class NLI:
    def __init__(self,root:Path):
        folder=root/'models/nli-deberta-small-int8'
        self.tokenizer=Tokenizer.from_file(str(folder/'tokenizer.json'))
        self.tokenizer.enable_truncation(512,strategy='only_first')
        self.tokenizer.enable_padding()
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(folder/'onnx/model_quantized.onnx'),options,providers=['CPUExecutionProvider'])
        self.labels=json.loads((folder/'config.json').read_text(encoding='utf-8'))['id2label']
        self.truncation_fallbacks=0

    def predict(self,pairs,batch_size=8):
        result=[]
        for start in range(0,len(pairs),batch_size):
            try:
                batch=self.tokenizer.encode_batch(pairs[start:start+batch_size])
            except Exception as error:
                if 'Truncation error' not in str(error):raise
                # Unbroken code/URL runs can exceed512 even in a96-cl100k
                # source unit. Keep the normal policy, record rare fallback.
                self.tokenizer.enable_truncation(512,strategy='longest_first')
                try:batch=self.tokenizer.encode_batch(pairs[start:start+batch_size])
                finally:self.tokenizer.enable_truncation(512,strategy='only_first')
                self.truncation_fallbacks+=1
            logits=self.session.run(None,{'input_ids':np.asarray([x.ids for x in batch],np.int64),
                                         'attention_mask':np.asarray([x.attention_mask for x in batch],np.int64)})[0]
            prob=np.exp(logits-logits.max(axis=1,keepdims=True));prob/=prob.sum(axis=1,keepdims=True)
            result.extend({self.labels[str(i)]:float(v) for i,v in enumerate(row)} for row in prob)
        return result
