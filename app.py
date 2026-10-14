"""
PDF Topic Extractor + Structurer
Powered by Qwen2.5-1.5B-Instruct with fine-tuned LoRA adapters (Extractor & Structurer).
"""

import os
import re
import json
import argparse
import torch
from pypdf import PdfReader
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login, HfApi
import gradio as gr

# Prompts & configuration
EXTRACT_SYSTEM = (
    "Extract from the text only the passages relevant to the given topic. "
    "Copy them verbatim, separated by blank lines. If nothing is relevant, reply exactly: NONE"
)
STRUCT_SYSTEM = (
    "Convert the extracted passages about a topic into structured JSON. Use only information in the passages. "
    "Output only a JSON object with keys: "
    "title (string), summary (2-3 sentences), key_points (list of 3-6 short strings), "
    "entities (list of {\"name\": string, \"type\": string}), dates (list of strings, empty if none). "
    "No text outside the JSON."
)
KEYS = {"title", "summary", "key_points", "entities", "dates"}
STRUCT_GROUP_CHARS = 1800


class PDFPipeline:
    def __init__(self, base_model: str, extractor_adapter: str, structurer_adapter: str, hf_token: str = None):
        if hf_token:
            login(token=hf_token)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.extract_bs = 8 if self.device == "cuda" else 1
        self.extract_max_tokens = 1000

        print(f"Loading tokenizer & base model '{base_model}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
        base = AutoModelForCausalLM.from_pretrained(base_model, dtype=self.dtype).to(self.device)

        print(f"Loading extractor adapter '{extractor_adapter}'...")
        self.model = PeftModel.from_pretrained(base, extractor_adapter, adapter_name="extractor")
        
        print(f"Loading structurer adapter '{structurer_adapter}'...")
        self.model.load_adapter(structurer_adapter, adapter_name="structurer")
        self.model.eval()
        print(f"Model and adapters successfully loaded on {self.device}.")

    def prompt(self, system: str, user: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True,
            tokenize=False,
        )

    @torch.no_grad()
    def generate(self, adapter: str, prompts: list, max_new_tokens: int):
        self.model.set_adapter(adapter)
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        n = enc["input_ids"].shape[1]
        return [self.tokenizer.decode(o[n:], skip_special_tokens=True).strip() for o in out]

    @staticmethod
    def is_none(s: str) -> bool:
        return s.strip().rstrip(".").strip().upper() == "NONE"

    @staticmethod
    def read_pdf(path: str) -> str:
        return "\n\n".join((p.extract_text() or "") for p in PdfReader(path).pages)

    @staticmethod
    def chunk_text(text: str, max_chars: int = 3000) -> list:
        paras = []
        for p in re.split(r"\n\s*\n", text):
            p = re.sub(r"\s*\n\s*", " ", p).strip()
            if not p:
                continue
            while len(p) > 1500:  # split very long paragraphs at a sentence boundary
                cut = p.rfind(". ", 0, 1500)
                cut = cut + 1 if cut > 300 else 1500
                paras.append(p[:cut].strip())
                p = p[cut:].strip()
            paras.append(p)
        chunks, cur = [], ""
        for p in paras:
            if cur and len(cur) + len(p) + 2 > max_chars:
                chunks.append(cur)
                cur = p
            else:
                cur = f"{cur}\n\n{p}" if cur else p
        if cur:
            chunks.append(cur)
        return chunks

    @staticmethod
    def parse_json(txt: str):
        try:
            start = txt.index("{")
            end = txt.rindex("}") + 1
            obj = json.loads(txt[start:end])
            return obj if KEYS <= set(obj) else {"error": "missing keys", "raw": txt}
        except Exception:
            return {"error": "invalid json", "raw": txt}

    @staticmethod
    def merge(results: list):
        ok = [r for r in results if "error" not in r]
        if not ok:
            return {"error": "structurer returned invalid JSON", "raw": results}
        uniq = lambda items, key: list({key(i): i for i in items}.values())
        return {
            "title": ok[0]["title"],
            "summary": " ".join(r["summary"] for r in ok),
            "key_points": uniq([k for r in ok for k in r["key_points"]], lambda k: str(k).lower()),
            "entities": uniq(
                [e for r in ok for e in r["entities"]],
                lambda e: str(e.get("name", "")).lower() if isinstance(e, dict) else str(e).lower(),
            ),
            "dates": uniq([d for r in ok for d in r["dates"]], lambda d: str(d).lower()),
        }

    def run(self, pdf_path: str, topic: str, progress=None):
        tick = progress or (lambda *a, **k: None)
        text = self.read_pdf(pdf_path)
        if not text.strip():
            return "No extractable text found (scanned PDF?).", None, ""

        chunks = self.chunk_text(text)
        found = []
        for i in range(0, len(chunks), self.extract_bs):
            tick(i / len(chunks), desc=f"Extracting {i}/{len(chunks)} chunks")
            outs = self.generate(
                "extractor",
                [self.prompt(EXTRACT_SYSTEM, f"Topic: {topic}\n\nText:\n{c}") for c in chunks[i:i + self.extract_bs]],
                self.extract_max_tokens,
            )
            found += [o for o in outs if o and not self.is_none(o)]
        found = list(dict.fromkeys(found))
        status = f"{len(chunks)} chunks, {len(found)} with relevant content"
        if not found:
            return status, {"message": "No content related to this topic found in the PDF."}, ""

        groups, cur = [], ""
        for f in found:
            if cur and len(cur) + len(f) > STRUCT_GROUP_CHARS:
                groups.append(cur)
                cur = f
            else:
                cur = f"{cur}\n\n{f}" if cur else f
        groups.append(cur)

        results = []
        for j, g in enumerate(groups):
            tick(0.9, desc=f"Structuring {j + 1}/{len(groups)}")
            out = self.generate("structurer", [self.prompt(STRUCT_SYSTEM, f"Topic: {topic}\n\nPassages:\n{g}")], 450)[0]
            results.append(self.parse_json(out))

        return status, self.merge(results), "\n\n---\n\n".join(found)


def launch_ui(pipeline: PDFPipeline, share: bool = False, server_port: int = 7860):
    def ui(pdf, topic, progress=gr.Progress()):
        if pdf is None or not topic.strip():
            return "Upload a PDF and enter a topic.", None, ""
        return pipeline.run(pdf, topic.strip(), progress)

    demo = gr.Interface(
        fn=ui,
        inputs=[
            gr.File(label="PDF", file_types=[".pdf"], type="filepath"),
            gr.Textbox(label="Topic", placeholder="e.g. quarterly revenue, machine learning methodology"),
        ],
        outputs=[
            gr.Textbox(label="Status"),
            gr.JSON(label="Structured output"),
            gr.Textbox(label="Extracted passages", lines=12),
        ],
        title="PDF Topic Extractor + Structurer",
        flagging_mode="never",
    )
    demo.launch(share=share, server_port=server_port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Topic Extractor and Structurer")
    parser.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model identifier")
    parser.add_argument("--extractor", default=None, help="Extractor LoRA adapter model")
    parser.add_argument("--structurer", default=None, help="Structurer LoRA adapter model")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"), help="Hugging Face API token")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--port", type=int, default=7860, help="Server port for Gradio")
    args = parser.parse_args()

    token = args.hf_token
    extractor = args.extractor
    structurer = args.structurer

    if not extractor or not structurer:
        if token:
            try:
                hf_user = HfApi(token=token).whoami()["name"]
                extractor = extractor or f"{hf_user}/pdf-extractor-qwen-1.5b"
                structurer = structurer or f"{hf_user}/pdf-structurer-qwen-1.5b"
            except Exception as e:
                print(f"Warning: Could not fetch HF username from token: {e}")
        
        if not extractor or not structurer:
            raise ValueError(
                "Please specify --extractor and --structurer adapter paths or provide HF_TOKEN environment variable."
            )

    pipeline = PDFPipeline(
        base_model=args.base,
        extractor_adapter=extractor,
        structurer_adapter=structurer,
        hf_token=token,
    )
    launch_ui(pipeline, share=args.share, server_port=args.port)
