"""Future extraction policy; never rewrites the authoritative frozen cache."""
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

class ColdRecordEncoder:
    policy='cold_record_ordered24_v1'
    def __init__(self,root):
        folder=Path(root)/'models/bge-small-int8';self.tokenizer=Tokenizer.from_file(str(folder/'tokenizer.json'))
        self.tokenizer.enable_truncation(256);self.tokenizer.enable_padding()
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(folder/'onnx/model_quantized.onnx'),options,providers=['CPUExecutionProvider']);self.new_count=0
    def encode(self,texts,query=False):
        texts=[('Represent this sentence for searching relevant passages: '+s) if query else s for s in texts]
        out=[]
        for start in range(0,len(texts),24):
            batch=self.tokenizer.encode_batch(texts[start:start+24])
            values={'input_ids':np.asarray([x.ids for x in batch],np.int64),'attention_mask':np.asarray([x.attention_mask for x in batch],np.int64),'token_type_ids':np.asarray([x.type_ids for x in batch],np.int64)}
            feed={x.name:values[x.name] for x in self.session.get_inputs()};z=self.session.run(None,feed)[0]
            z=z[:,0,:] if z.ndim==3 else z;z=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12);out.append(z.astype(np.float32))
        self.new_count+=len(texts);return np.concatenate(out)
