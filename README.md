# AutoImpress: LLM-Based Radiology Impression Generator

![AutoImpress](https://img.shields.io/badge/AutoImpress-Clinical_AI-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=flat-square)
![Transformers](https://img.shields.io/badge/🤗_Transformers-4.21.0-yellow?style=flat-square)

*Automated clinical impression generation from structured radiology reports using Large Language Models*

**[📊 Results](#-key-results) • [🚀 Quick Start](#-quick-start) • [📖 Documentation](#-project-structure) • [🎯 Architecture](#-methodology)**

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [👥 Team](#-team)
- [🏥 Dataset](#-dataset)
- [🧠 Methodology](#-methodology)
- [📊 Key Results](#-key-results)
- [📦 Project Structure](#-project-structure)
- [🚀 Quick Start](#-quick-start)
- [📈 Advanced Analysis](#-advanced-analysis)
- [💡 Key Insights](#-key-insights)
- [🔬 References](#-references)
---

## 🎯 Overview

AutoImpress is a comprehensive research project that explores **automated clinical impression generation** from structured radiology report fields using state-of-the-art Large Language Models (LLMs). The project benchmarks multiple approaches including fine-tuning and prompt engineering on the **IU-XRay dataset**.

## 🖼️ Graphical Abstract

![AutoImpress Graphical Abstract](images/autoimpress_graphical_abstract.png)

### 🎪 Problem Statement
- **Input**: Structured radiology report fields (Findings, Indication, Comparison, etc.)
- **Output**: Concise, clinically accurate impression summaries
- **Challenge**: Bridging the gap between surface-level text similarity and clinical equivalence

### 🏆 Key Contributions
- Novel evaluation framework using GPT-4o clinical judgment
- In-depth analysis of model agreement and disagreement patterns
- Practical insights for medical NLP applications

---

## 👥 Team

| Team Member | 
|-------------|
| **Yaniv Grosberg** |
| **Netanel Ohev Shalom** |
| **Aviel Shmuel** | 

---

## 🏥 Dataset

### IU-XRay (Indiana University Chest X-ray Reports)
- **Original**: 3,955 radiology reports
- **After Preprocessing**: 3,331 records (~13% reduction)
- **Fields**: Findings, Indication, Comparison, Image, MeSH, Problems → **Impression**

### 📊 Dataset Statistics
- **Text Length**: Findings ≈ 190 chars, Impression ≈ 76 chars
- **Vocabulary**: ~2,000 unique tokens
- **Common Pattern**: 

| Phrase                                      | Count |
|--------------------------------------------|-------|
| No acute cardiopulmonary abnormality        | 491   |
| No acute cardiopulmonary findings           | 189   |
| No acute cardiopulmonary abnormalities      | 168   |
| No acute cardiopulmonary disease            | 163   |
| No acute disease                            | 126   |
| No acute cardiopulmonary process            | 106   |
| No acute radiographic cardiopulmonary process | 93  |
| No acute cardiopulmonary abnormality identified | 80 |
| No acute pulmonary disease                  | 76    |
| No acute findings                           | 60    |


### 🔧 Preprocessing Steps
- Removed ~520 incomplete records
- Filled missing fields with "none provided"
- Replaced anonymized patterns with `[REDACTED]`
- Cleaned non-clinical tokens for prompt optimization

---

## 🧠 Methodology

### 🤖 Models Evaluated

| Model                 | Type        | Configuration              | Platform             |
|-----------------------|------------|----------------------------|----------------------|
| **FLAN-T5 Base**      | Baseline   | Few-shot prompting         | Google Colab (GPU)   |
| **FLAN-T5 Fine-tuned**| Fine-tuned | 3 epochs, batch size 4     | Google Colab (GPU)   |
| **GPT-4.1**           | API-based  | Prompt engineering         | Azure API            |
| **DeepSeek-V3**       | API-based  | Prompt engineering         | Azure API            |


### 🔄 Processing Pipelines

#### FLAN-T5 Pipeline
```
Data → Few-shot Baseline → Fine-tune → Validation → GPT-4o Judge
```

#### GPT-4.1 & DeepSeek Pipeline
```
Data → 50% Sample → API Generation → BERTScore → GPT-4o Judge
```

### 📏 Evaluation Metrics

#### Primary Metric: GPT-4o Clinical Judge
- **Approach**: Binary YES/NO clinical equivalence assessment
- **Prompt**: Expert medical comparison with strict clinical criteria
- **Focus**: Clinical interpretability over surface similarity

#### Secondary Metric: BERTScore F1
- **Purpose**: Semantic similarity measurement
- **Range**: 0-1 (higher = more similar)
- **Limitation**: May not capture clinical nuances

---

## 📊 Key Results

### 🏆 Model Performance Summary

| Model                 | Clinical Equivalence        | BERTScore F1           | Samples |
|-----------------------|----------------------------|------------------------|---------|
| **FLAN-T5 Baseline**  | **1.3%** (4/300)          | 0.8382 ± 0.0281        | 300     |
| **FLAN-T5 Fine-Tuned**| **47.7%** (318/667)       | 0.9227 ± 0.0536        | 667     |
| **GPT-4.1**           | **77.1%** (1284/1666)     | 0.8794 ± 0.0317        | 1666    |
| **DeepSeek**          | **78.6%** (1309/1666)     | 0.8826 ± 0.0319        | 1666    |


### 📈 Performance Insights

#### ✅ Success Stories
- **Fine-tuning Impact**: Massive improvement from 1.3% → 47.7% clinical equivalence
- **Large Model Excellence**: GPT-4.1 and DeepSeek achieved near-80% clinical accuracy
- **High Agreement**: 83.7% agreement between GPT-4.1 and DeepSeek
- **Prompt Engineering Success**: Multiple refinement rounds yielded optimal formulations

#### 🔍 Key Findings
- **129 Flipped Cases**: Fine-tuning converted 129/300 "No" → "Yes" judgments
- **Pattern Recognition**: Models performed differently on "no acute" vs. complex cases
- **Clinical vs. Semantic**: High BERTScore didn't guarantee clinical equivalence

---

## 📦 Project Structure

```
AutoImpress/
├── 📁 data_raw/                     # Raw datasets
│   └── indiana_reports.csv
├── 📁 data_cleaned/                 # Processed datasets  
│   └── indiana_reports_cleaned.csv
├── 📁 outputs/
│   ├── 📁 flan-t5-base/            # FLAN-T5 results
│   │   ├── generated_impressions_300_flan.csv
│   │   ├── finetuned_model_test_results.csv
│   │   ├── results_with_azure_gpt_judgment_baseline.csv
│   │   └── results_with_azure_gpt_judgment.csv
│   ├── 📁 gpt-deepseek/            # GPT-4.1 & DeepSeek results
│   │   ├── gpt41_judged_results.csv
│   │   ├── deepseek_judged_results.csv
│   │   ├── gpt4_1_acute_findings_vs_ground_truth.csv
│   │   └── deepseek_acute_findings_vs_ground_truth.csv
│   └── 📁 Analyze_Results_outputs/  # Analysis outputs
│       └── judgment_phrase_summary.csv
├── 📁 notebooks/
│   ├── 📓 Preprocessing_EDA.ipynb   # Data exploration & cleaning
│   ├── 📓 flan-t5-base.ipynb       # FLAN-T5 training & evaluation
│   ├── 📓 gpt_deepseek.ipynb       # API-based model evaluation
│   └── 📓 Analyze_Results.ipynb    # Comprehensive results analysis
├── 🐍 utils_file.py                # Shared utility functions
├── 📋 requirements.txt             # Python dependencies
└── 📖 README.md                    # Project documentation
```

### 📊 Output File Meanings
- **`uid`**: Unique identifier for each report
- **`generated_impression`**: Model-generated clinical summary
- **`true_impression`**: Expert-labeled ground truth
- **`gpt_equivalence`**: GPT-4o clinical judgment (Yes/No)

---

## 🚀 Quick Start

### 1️⃣ Clone Repository
```bash
git clone https://github.com/Yanivgg/AutoImpress.git
cd AutoImpress
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run Analysis Pipeline
```bash
# Data preprocessing and EDA
jupyter notebook notebooks/Preprocessing_EDA.ipynb

# FLAN-T5 model training and evaluation
jupyter notebook notebooks/flan-t5-base.ipynb

# GPT-4.1 and DeepSeek evaluation
jupyter notebook notebooks/gpt_deepseek.ipynb

# Comprehensive results analysis
jupyter notebook notebooks/Analyze_Results.ipynb
```

### 4️⃣ Explore Results
Generated outputs will be saved in the `outputs/` directory with detailed CSV files for further analysis.

---

## 📈 Advanced Analysis

### 🤝 Model Agreement Analysis
- **Overall Agreement**: GPT-4.1 and DeepSeek agreed in **83.7%** of cases
- **Disagreement Cases**: 271 instances (16.3%) revealed interesting edge cases
- **Fine-tuning Impact**: 129 cases improved from baseline to fine-tuned FLAN-T5

### 🔍 Pattern Recognition
Models showed varying performance on different clinical phrases:

| Phrase Type         | FLAN Baseline | FLAN Fine-tuned | GPT-4.1 | DeepSeek |
|---------------------|---------------|-----------------|---------|----------|
| "No acute" patterns | ~0%           | ~75%            | ~85%    | ~85%     |
| Complex findings    | ~0%           | ~25%            | ~70%    | ~75%     |



### 📊 Clinical Phrase Analysis
- **"No acute cardiopulmonary abnormality"**: Highest agreement across all models
- **Complex multi-finding cases**: Greater model divergence
- **Edge cases**: Models disagreed on borderline clinical interpretations

---

## 💡 Key Insights

### 🎯 Technical Insights
1. **Fine-tuning is Crucial**: Baseline FLAN-T5 performed poorly (1.3%) but fine-tuning achieved 47.7%
2. **Large Models Excel**: GPT-4.1 and DeepSeek reached ~78% without domain-specific training
3. **Prompt Engineering Matters**: Multiple refinement rounds were essential for optimal performance
4. **Clinical ≠ Semantic**: High BERTScore doesn't guarantee clinical equivalence

### 🏥 Clinical Implications
1. **Pattern Templates**: Dataset's repetitive nature aids prompt steering but may limit generalization
2. **"No Acute" Advantage**: Models perform better on normal findings vs. complex pathology
3. **Human Validation Essential**: Even best models miss ~22% of clinical nuances
4. **Structured Input Benefits**: Organized report fields improve generation quality

### 🔬 Research Contributions
1. **Evaluation Framework**: Novel use of GPT-4o for clinical judgment evaluation
2. **Comprehensive Benchmarking**: Systematic comparison across model types and sizes
3. **Practical Insights**: Real-world applicability analysis for medical NLP systems
4. **Open Science**: Full reproducible pipeline with detailed analysis

---

## 🔬 References

### 📚 Key Papers
- **Zhang et al. (2023)**: *Leveraging Summary Guidance on Medical Report Summarization* - Fine-tuning BART/T5 on medical reports, achieved superior performance on DISCHARGE and ECHO datasets
- **Ma et al. (2023)**: *From General to Specific: Domain Adaptation for Medical Report Generation* - HybridFusion approach combining general and medical LLMs, SOTA results on MIMIC-CXR
- **Van Veen et al. (2023)**: *RadAdapt: Lightweight Domain Adaptation of LLMs* - Efficient LoRA/Prefix tuning achieving clinical validation with minimal parameters

### 🛠️ Technical Stack
- **Models**: FLAN-T5, GPT-4.1, DeepSeek-V3
- **Evaluation**: BERTScore, GPT-4o Clinical Judge
- **Frameworks**: Transformers, PyTorch, Azure OpenAI API
- **Environment**: Google Colab Pro, VSCode, Jupyter Notebooks


## 🙏 Acknowledgments

- **Indiana University** for the IU-XRay dataset
- **Microsoft Azure** for providing free student accounts that enabled API access
- **Holon Institute Of Technology Academic Institution** for supporting this research project

---


**⭐ Star this repository if you find it useful! ⭐**

*For questions, issues, or collaboration opportunities, please open an issue or reach out to the team.*

