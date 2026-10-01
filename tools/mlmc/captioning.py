"""Image captioning with small vision-language models (ONNX, greedy decoding).

Every captioning runner returns ``Caption``: the generated text and its
token count. Decoding is greedy with a KV cache, implemented with plain
ONNX Runtime and the ``tokenizers`` package (no transformers at run time).

* ``Florence2`` — encoder-decoder (onnx-community/Florence-2-*): vision
  encoder -> [image features ; prompt embeddings] -> BART-style encoder ->
  merged decoder with past key/values (``use_cache_branch``).
* ``SmolVLM`` — decoder-only (HuggingFaceTB/SmolVLM-*-Instruct official
  ONNX): image features replace the 64 ``<image>`` token embeddings of the
  chat prompt, then a merged decoder with past key/values and position ids.

The benchmark's model-only latency is the vision encoder; end-to-end is the
whole caption (all decoding steps).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from tools.mlmc.detection import ort_session

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class Caption:
    text: str
    tokens: int


def _tokenizer(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))


class _Base:
    def __init__(self, model, provider="cuda", max_new_tokens=None):
        cfg = model.meta["captioning"]
        wd = model.weights_dir
        self.sess = ort_session(model.artifact_path("onnx"), provider)  # vision encoder
        self.embed = ort_session(wd / cfg["embed_file"], provider)
        self.decoder = ort_session(wd / cfg["decoder_file"], provider)
        self.tok = _tokenizer(wd / cfg["tokenizer_file"])
        self.cfg = cfg
        self.max_new = max_new_tokens or cfg.get("max_new_tokens", 40)
        self.past_names = [i.name for i in self.decoder.get_inputs() if i.name.startswith("past_key_values")]

    def infer(self, x):
        return self.sess.run(None, {i.name: v for i, v in zip(self.sess.get_inputs(), x)})[0]

    def emb(self, ids):
        return self.embed.run(None, {"input_ids": np.asarray(ids, np.int64)[None]})[0]

    def __call__(self, frame_bgr):
        x, meta = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), meta)


class Florence2(_Base):
    def __init__(self, model, provider="cuda", max_new_tokens=None):
        super().__init__(model, provider, max_new_tokens)
        self.encoder = ort_session(model.weights_dir / self.cfg["encoder_file"], provider)
        self.size = self.cfg["image_size"]
        self.prompt_ids = self.tok.encode(self.cfg["prompt"]).ids  # <s> ... </s>

    def preprocess(self, frame_bgr):
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).resize((self.size, self.size), Image.BICUBIC)
        x = (np.asarray(img, np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return (np.ascontiguousarray(x.transpose(2, 0, 1)[None]),), None

    def postprocess(self, image_features, meta=None):
        enc_in = np.concatenate([image_features, self.emb(self.prompt_ids)], axis=1)
        mask = np.ones(enc_in.shape[:2], np.int64)
        enc = self.encoder.run(None, {"inputs_embeds": enc_in, "attention_mask": mask})[0]
        past = {n: np.zeros((1, 12, 0, 64), np.float32) for n in self.past_names}
        tok, out = 2, []  # decoder_start_token_id
        enc_kv = None
        for step in range(self.max_new):
            feeds = {"inputs_embeds": self.emb([tok]), "encoder_hidden_states": enc,
                     "encoder_attention_mask": mask, "use_cache_branch": np.array([step > 0]), **past}
            res = self.decoder.run(None, feeds)
            names = [o.name for o in self.decoder.get_outputs()]
            logits, presents = res[0], dict(zip(names[1:], res[1:]))
            if enc_kv is None:  # cross-attention keys / values come from the first step
                enc_kv = {k: v for k, v in presents.items() if ".encoder." in k}
            past = {n: (enc_kv if ".encoder." in n else presents)[n.replace("past_key_values", "present")]
                    for n in self.past_names}
            tok = int(logits[0, -1].argmax())
            if tok == 2:  # eos
                break
            out.append(tok)
        return Caption(self.tok.decode(out, skip_special_tokens=True).strip(), len(out))


class SmolVLM(_Base):
    IMAGE_TOKEN = "<image>"

    def __init__(self, model, provider="cuda", max_new_tokens=None):
        super().__init__(model, provider, max_new_tokens)
        self.size = self.cfg["image_size"]
        n_img = self.cfg["image_tokens"]
        prompt = ("<|im_start|>User:<fake_token_around_image><global-img>" + self.IMAGE_TOKEN * n_img
                  + "<fake_token_around_image>" + self.cfg["prompt"] + "<end_of_utterance>\nAssistant:")
        self.prompt_ids = self.tok.encode(prompt, add_special_tokens=False).ids
        self.image_id = self.tok.token_to_id(self.IMAGE_TOKEN)
        self.stop = {self.tok.token_to_id("<end_of_utterance>"), self.tok.token_to_id("<|im_end|>")}
        kv = next(i for i in self.decoder.get_inputs() if i.name.startswith("past_key_values"))
        self.kv_heads, self.head_dim = kv.shape[1], kv.shape[3]
        self.kv_dtype = np.float16 if "float16" in kv.type else np.float32

    def preprocess(self, frame_bgr):
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).resize((self.size, self.size), Image.BILINEAR)
        x = (np.asarray(img, np.float32) / 255.0 - 0.5) / 0.5
        px = np.ascontiguousarray(x.transpose(2, 0, 1)[None, None])
        return (px, np.ones((1, 1, self.size, self.size), bool)), None

    def postprocess(self, image_features, meta=None):
        embeds = self.emb(self.prompt_ids)
        pos = np.flatnonzero(np.array(self.prompt_ids) == self.image_id)
        embeds[0, pos] = image_features.reshape(-1, embeds.shape[-1])[:len(pos)]
        past = {n: np.zeros((1, self.kv_heads, 0, self.head_dim), self.kv_dtype) for n in self.past_names}
        S = embeds.shape[1]
        feeds_embeds, position = embeds, np.arange(S, dtype=np.int64)[None]
        total, out = S, []
        for _ in range(self.max_new):
            res = self.decoder.run(None, {"inputs_embeds": feeds_embeds, "attention_mask": np.ones((1, total), np.int64),
                                          "position_ids": position, **past})
            names = [o.name for o in self.decoder.get_outputs()]
            presents = dict(zip(names[1:], res[1:]))
            past = {n: presents[n.replace("past_key_values", "present")] for n in self.past_names}
            tok = int(res[0][0, -1].argmax())
            if tok in self.stop:
                break
            out.append(tok)
            feeds_embeds = self.emb([tok])
            position = np.array([[total]], np.int64)
            total += 1
        return Caption(self.tok.decode(out, skip_special_tokens=True).strip(), len(out))


def render(frame: np.ndarray, cap: Caption, width_chars: int = 46) -> np.ndarray:
    """Caption wrapped in a band at the bottom of the frame."""
    import textwrap
    out = frame.copy()
    H, W = out.shape[:2]
    s = W / 640
    lines = textwrap.wrap(cap.text, width_chars)[:4] or ["(empty)"]
    lh = int(30 * s)
    top = H - lh * len(lines) - int(16 * s)
    out[top:] = (out[top:] * 0.25).astype(np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(out, t, (int(10 * s), top + lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s,
                    (255, 255, 255), max(1, round(2 * s)), cv2.LINE_AA)
    return out
