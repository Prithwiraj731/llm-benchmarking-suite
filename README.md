<div align="center">

# 🚗 Fine-Tuned LLM Performance Benchmarking Suite

### *Comprehensive Evaluation Framework for Domain-Specific Language Models*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/docs/transformers)
[![Google Colab](https://img.shields.io/badge/Colab-Optimized-orange.svg)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Prithwiraj731/llm-benchmarking-suite)

*A production-ready evaluation suite for comparing fine-tuned Gemma-2-2B models on vehicle manual Q&A tasks*

[Features](#-features) • [Models](#-models-evaluated) • [Metrics](#-evaluation-metrics) • [Quick Start](#-quick-start) • [Results](#-results) • [Documentation](#-documentation)

---

</div>

## 📋 Overview

This repository contains a **comprehensive benchmarking framework** for evaluating and comparing fine-tuned Large Language Models (LLMs) on domain-specific question-answering tasks. The project demonstrates professional ML evaluation practices with **5 distinct metric categories** across **2 fine-tuned Gemma-2-2B models**.

### 🎯 Project Highlights

- ✅ **Production-Ready Evaluation Scripts** - Optimized for Google Colab with 4-bit quantization
- 📊 **5 Comprehensive Metric Categories** - BERT Score, Task Accuracy, BLEU/ROUGE-L, Latency, Memory
- 🔬 **Rigorous Testing Protocol** - Temperature 0.1, greedy decoding, consistent prompt formatting
- 📈 **Visual Analytics** - Automated chart generation for all metrics
- 🚀 **Reproducible Results** - Detailed notebooks with step-by-step execution
- 🎓 **Educational Resource** - Learn best practices for LLM evaluation

---

## 🤖 Models Evaluated

<table>
<tr>
<th>Model</th>
<th>Type</th>
<th>Domain</th>
<th>HuggingFace</th>
</tr>
<tr>
<td><b>🏍️ Two-Wheeler BSA</b></td>
<td>LoRA Adapter</td>
<td>BSA Motorcycle Manual</td>
<td><a href="https://huggingface.co/Prithwiraj731/Gemma2-2b_Two-Wheeler">View Model</a></td>
</tr>
<tr>
<td><b>🚗 Four-Wheeler Lexus</b></td>
<td>Full Merged</td>
<td>Lexus Vehicle Manual</td>
<td><a href="https://huggingface.co/Prithwiraj731/FourWheeler-Gemma-2B">View Model</a></td>
</tr>
</table>

**Base Model:** `google/gemma-2-2b` | **Quantization:** 4-bit NF4 | **Framework:** Transformers + PEFT

---

## 📊 Evaluation Metrics

### 1️⃣ **BERT Score** - Semantic Similarity
Measures semantic similarity between generated and reference answers using contextual embeddings.

- **Metrics:** Precision, Recall, F1 Score
- **Range:** -1.0 (worst) to 1.0 (best)
- **Baseline:** `microsoft/deberta-xlarge-mnli`

### 2️⃣ **Task Accuracy** - Answer Correctness
Evaluates factual correctness through multiple matching strategies.

- **Exact Match:** Character-level exact matching (normalized)
- **Partial Match:** ≥30% word overlap threshold
- **Keyword Score:** Non-stopword matching percentage

### 3️⃣ **BLEU & ROUGE-L** - N-gram Overlap
Standard machine translation metrics adapted for Q&A evaluation.

- **BLEU-1/2/4:** Unigram, bigram, and 4-gram precision
- **ROUGE-L:** Longest common subsequence F-measure
- **Smoothing:** Method 4 (Chen & Cherry)

### 4️⃣ **Inference Latency** - Performance Metrics
Detailed timing analysis for production deployment planning.

- **Tokenization Time:** Input preprocessing latency
- **Inference Time:** Model forward pass duration
- **Decoding Time:** Output generation overhead
- **Throughput:** Tokens generated per second
- **Percentiles:** P50, P90, P99 latency distribution

### 5️⃣ **Memory Footprint** - Resource Utilization
GPU memory profiling for infrastructure cost estimation.

- **Model Size:** Parameters + buffers in memory
- **GPU Allocation:** Peak and reserved memory
- **Memory Breakdown:** Per-stage memory consumption
- **Parameter Count:** Total vs. trainable parameters

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
# CUDA-compatible GPU (recommended)
# Google Colab account (for cloud execution)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Prithwiraj731/llm-benchmarking-suite.git
cd llm-benchmarking-suite

# Install dependencies
pip install -q transformers accelerate bitsandbytes peft bert-score rouge-score nltk
```

### Running Evaluations

#### Option 1: Google Colab (Recommended)

1. Upload notebooks to Google Colab
2. Enable GPU runtime: `Runtime → Change runtime type → GPU`
3. Run cells sequentially (restart runtime after package installation)

#### Option 2: Local Execution

```python
# Example: BERT Score Evaluation
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from bert_score import score

# Load model
tokenizer = AutoTokenizer.from_pretrained("Prithwiraj731/Gemma2-2b_Two-Wheeler")
model = PeftModel.from_pretrained(base_model, "Prithwiraj731/Gemma2-2b_Two-Wheeler")

# Run evaluation
# See notebooks for complete implementation
```

---

## 📁 Repository Structure

```
📦 Metric Calculations
├── 🏍️ Two Wheeler Benchmarks/
│   ├── Two Wheeler BERT Scores/
│   │   ├── Two_Wheeler_BERT.ipynb
│   │   └── Two_Wheeler_BERT.png
│   ├── Task Accuracy/
│   │   ├── Task_Accuracy.ipynb
│   │   └── Task_Accuracy.png
│   ├── BLEU_ROUGE-L Scores/
│   │   ├── Bleu_Rouge_L.ipynb
│   │   └── Bleu_Rouge_L.png
│   ├── Inference Latency/
│   │   ├── inference_latency_2wheeler.ipynb
│   │   └── inference_latency_2wheeler.png
│   └── Memory Footprint/
│       ├── memory_footprint_2wheeler.ipynb
│       ├── memory_footprint_2wheeler.png
│       └── DETAILED_MEMORY_USAGE_TABLE.png
│
├── 🚗 Four Wheeler Benchmarks/
│   ├── Four Wheeler Bert Scores/
│   │   ├── Four_Wheeler_BERT.ipynb
│   │   └── Four_Wheeler_BERT.png
│   ├── Task Accuracy/
│   │   ├── Task_Accuracy.ipynb
│   │   └── Task_Accuracy.png
│   ├── BLEU_ROUGE-L Scores/
│   │   ├── Bleu_Rouge-L.ipynb
│   │   └── Bleu_Rouge-L.png
│   ├── Inference Latency/
│   │   ├── inference_latency_4wheeler.ipynb
│   │   └── inference_latency_4wheeler.png
│   └── Memory Footprint/
│       ├── memory_footprint_4wheeler.ipynb
│       ├── memory_footprint_4wheeler.png
│       └── DETAILED_MEMORY_USAGE_TABLE.png
│
└── 📄 README.md
```

---

## 🎯 Results

> **Note:** Detailed results with visualizations are available in the respective benchmark folders. Each metric includes:
> - Per-question breakdown
> - Average scores
> - Statistical analysis
> - Visual charts

### Sample Evaluation Dataset

**Two-Wheeler (BSA):** 6 questions from BSA D14/4 Bantam Supreme manual
- Engine lubrication specifications
- Service department procedures
- Dealer consultation guidelines
- Technical specifications

**Four-Wheeler (Lexus):** 6 questions from Lexus owner's manual
- SRS airbag functionality
- Bluetooth connectivity procedures
- Overheating troubleshooting
- Parts replacement guidelines

---

## 🛠️ Technical Details

### Optimization Settings

```python
# Consistent across all evaluations
TEMPERATURE = 0.1          # Deterministic generation
DO_SAMPLE = False          # Greedy decoding
MAX_NEW_TOKENS = 100       # Response length limit
REPETITION_PENALTY = 1.2   # Prevent repetition
PROMPT_FORMAT = "simple"   # Question\n format
```

### Quantization Configuration

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
```

### Hardware Requirements

- **Minimum:** 8GB GPU VRAM (Google Colab Free Tier)
- **Recommended:** 16GB+ GPU VRAM for faster inference
- **CPU:** Multi-core processor for parallel processing
- **RAM:** 16GB+ system memory

---

## 📖 Documentation

### Notebooks Overview

Each notebook follows a standardized structure:

1. **Package Installation** - Dependencies and environment setup
2. **Library Imports** - Required modules
3. **Dataset Definition** - Test questions and reference answers
4. **Model Configuration** - Model loading and setup
5. **Evaluation Execution** - Metric calculation
6. **Results Visualization** - Charts and tables
7. **Export Options** - CSV download functionality

### Metric Interpretation Guides

**BERT Score Quality Thresholds:**
- 🟢 **0.7 - 1.0:** Excellent semantic similarity
- 🟡 **0.5 - 0.7:** Good similarity
- 🟠 **0.3 - 0.5:** Moderate similarity
- 🔴 **0.0 - 0.3:** Poor similarity

**Latency Benchmarks:**
- ⚡ **< 100ms:** Real-time capable
- ✅ **100-500ms:** Interactive applications
- ⚠️ **500-1000ms:** Acceptable for batch processing
- 🔴 **> 1000ms:** Optimization needed

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution

- 🆕 Additional evaluation metrics (perplexity, diversity scores)
- 📊 Enhanced visualization dashboards
- 🔧 Optimization techniques comparison
- 📝 Documentation improvements
- 🐛 Bug fixes and performance improvements

---

## 📚 Citation

If you use this benchmarking suite in your research or project, please cite:

```bibtex
@misc{llm-benchmarking-suite,
  author = {Prithwiraj Dutta},
  title = {Fine-Tuned LLM Performance Benchmarking Suite},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/Prithwiraj731/llm-benchmarking-suite}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Google** - Gemma-2-2B base model
- **Hugging Face** - Transformers library and model hosting
- **Microsoft** - DeBERTa for BERT Score baseline
- **NLTK & Rouge-Score** - Evaluation metric implementations
- **Google Colab** - Free GPU resources for development

---

## 📞 Contact & Support

- **Issues:** [GitHub Issues](https://github.com/Prithwiraj731/llm-benchmarking-suite/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Prithwiraj731/llm-benchmarking-suite/discussions)
- **Email:** prithwi1016@gmail.com

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

**Made with ❤️ for the AI/ML Community**

[⬆ Back to Top](#-fine-tuned-llm-performance-benchmarking-suite)

</div>
