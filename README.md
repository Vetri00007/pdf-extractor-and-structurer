# PDF Topic Extractor and Structurer

An end-to-end pipeline that extracts topic-specific passages from PDFs and synthesizes them into structured JSON using **Qwen2.5-1.5B-Instruct** with fine-tuned **LoRA adapters**.

---

## 🚀 Features

- **Document Ingestion**: Extracts text from PDFs and chunks it intelligently with sentence boundary preservation.
- **Topic Extraction**: Uses a dedicated fine-tuned LoRA adapter (`pdf-extractor-qwen-1.5b`) to pull verbatim passages relevant to your target topic.
- **Information Structuring**: Uses a second fine-tuned LoRA adapter (`pdf-structurer-qwen-1.5b`) to synthesize findings into structured JSON containing:
  - Title
  - Summary (2–3 sentences)
  - Key points
  - Named entities (`name`, `type`)
  - Mentioned dates
- **Interactive UI**: Gradio web interface with live progress bars and public sharing support.
- **Colab Ready**: Includes a Google Colab notebook pre-configured for free T4 GPU environments.

---

## 📁 Repository Structure

```
├── app.py                         # Standalone Python application & Gradio UI
├── pdf_extractor_app_colab.ipynb  # Interactive Google Colab notebook
├── requirements.txt               # Dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # Project documentation
```

---

## 🛠️ Quickstart

### Option 1: Google Colab (Recommended for fast GPU access)

1. Open [`pdf_extractor_app_colab.ipynb`](pdf_extractor_app_colab.ipynb) in Google Colab.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Add your Hugging Face token in Colab Secrets as `HF_TOKEN` (enable notebook access).
4. Run all cells to launch the Gradio public interface.

### Option 2: Local / Server Setup

#### 1. Clone the repository
```bash
git clone <YOUR_REPOSITORY_URL>
cd pdf-extractor-and-structurer
```

#### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install dependencies
```bash
pip install -r requirements.txt
```

#### 4. Run the application
Set your Hugging Face token and run `app.py`:

```bash
export HF_TOKEN="your_hf_token_here"
python app.py --share
```

Or pass custom adapter paths:
```bash
python app.py \
  --base "Qwen/Qwen2.5-1.5B-Instruct" \
  --extractor "your-hf-username/pdf-extractor-qwen-1.5b" \
  --structurer "your-hf-username/pdf-structurer-qwen-1.5b" \
  --hf-token "your_hf_token_here"
```

---

## ⚙️ Output Schema

The structurer produces JSON adhering to the following structure:

```json
{
  "title": "Topic or Document Title",
  "summary": "Concise 2-3 sentence overview of the topic in the document.",
  "key_points": [
    "Key finding or metric 1",
    "Key finding or metric 2"
  ],
  "entities": [
    {"name": "Entity Name", "type": "Organization/Person/Technology"}
  ],
  "dates": [
    "YYYY-MM-DD"
  ]
}
```

---

## 📄 License

MIT License. See individual model weights and datasets for their respective licensing terms.
