"""Pinned BGE-small ONNX CPU features, reusable across all selectors."""
from pathlib import Path
import hashlib
import sqlite3
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

REVISION = "ea104dacec62c0de699686887e3f920caeb4f3e3"


class Encoder:
    def __init__(self, root: Path):
        model = root / "models/bge-small-int8"
        self.tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model / "onnx/model_quantized.onnx"), options,
                                           providers=["CPUExecutionProvider"])
        dest = root / "data/cache"
        dest.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(dest / "bge_small_embeddings.sqlite3")
        self.db.execute("CREATE TABLE IF NOT EXISTS vec (id TEXT PRIMARY KEY, value BLOB)")
        self.new_count = 0

    def encode(self, texts, query=False):
        if query:
            texts = ["Represent this sentence for searching relevant passages: " + s for s in texts]
        keys = [hashlib.sha256((REVISION + "|cls|256|" + s).encode()).hexdigest() for s in texts]
        result = [None] * len(texts)
        missing = []
        for i,key in enumerate(keys):
            row = self.db.execute("SELECT value FROM vec WHERE id=?", (key,)).fetchone()
            if row:
                result[i] = np.frombuffer(row[0], dtype=np.float32).copy()
            else:
                missing.append(i)
        for start in range(0, len(missing), 24):
            ids = missing[start:start+24]
            batch = self.tokenizer.encode_batch([texts[i] for i in ids])
            values = {"input_ids":np.asarray([x.ids for x in batch],np.int64),
                      "attention_mask":np.asarray([x.attention_mask for x in batch],np.int64),
                      "token_type_ids":np.asarray([x.type_ids for x in batch],np.int64)}
            feed = {x.name:values[x.name] for x in self.session.get_inputs()}
            output = self.session.run(None, feed)[0]
            vectors = output[:,0,:] if output.ndim == 3 else output
            vectors = vectors / np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-12)
            for i,v in zip(ids,vectors):
                result[i] = v.astype(np.float32)
                self.db.execute("INSERT OR REPLACE INTO vec VALUES (?,?)",(keys[i],result[i].tobytes()))
            self.db.commit()
            self.new_count += len(ids)
        return np.stack(result)
