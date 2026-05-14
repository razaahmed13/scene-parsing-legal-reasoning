# Urban Scene Parsing & Agentic Legal Reasoning for Traffic Safety

**Author:** Ahmed Raza  

---

## Project Overview

This project builds an end-to-end intelligent traffic monitoring system in two parts:

- **Part 1** trains computer vision models (YOLOv11, SegFormer, U-Net) on the
  BDD100K urban driving dataset for object detection and semantic segmentation.
- **Part 2** builds an agentic RAG pipeline that reads YOLO detection logs,
  retrieves applicable Pakistani traffic law from a local FAISS vector database,
  and generates formal Citation Reports using a local LLM — no cloud APIs involved.

---

## Repository Structure

DLP_Project/
├── Part1/
│   ├── object_detection_with_YOLOv11.ipynb
│   └── semantic_segmentation_with_SegFormer.ipynb
├── Part2/
│   ├── agentic_RAG_based_legal_reasoning.ipynb
│   ├── detection_logs.json
│   └── law_documents/
│       └── pakistan_traffic_laws.txt
├── rag/
│   ├── law_index.faiss
│   ├── law_metadata.csv
├── evaluation/
│   ├── evaluation_rubric.csv
│   ├── evaluation_quantitative.csv
│   └── efficiency_analysis.csv
├── report/
│   └── report.pdf
├── requirements.txt
└── README.md

---

## Part 1 — Results Summary

### Object Detection — YOLOv11-n (BDD100K)

| Configuration     | Precision | Recall    | mAP@50    | mAP@50-95 | Epochs | Speed (ms/img) |
|-------------------|-----------|-----------|-----------|-----------|--------|----------------|
| No Augmentation   | 0.477     | 0.314     | 0.345     | 0.187     | 31     | 119.44         |
| Full Augmentation | **0.602** | **0.399** | **0.437** | **0.233** | 50     | **25.29**      |
| Δ (aug − no aug)  | +0.125    | +0.085    | +0.092    | +0.046    | —      | —              |

### Per-Class Detection Results (Augmented Run, 2,000 test images)

| Class         | Precision | Recall | mAP@50 | mAP@50-95 |
|---------------|-----------|--------|--------|-----------|
| car           | 0.710     | 0.655  | 0.705  | 0.420     |
| person        | 0.659     | 0.415  | 0.481  | 0.213     |
| traffic light | 0.594     | 0.468  | 0.473  | 0.154     |
| truck         | 0.615     | 0.454  | 0.511  | 0.353     |
| bus           | 0.587     | 0.447  | 0.491  | 0.372     |
| traffic sign  | 0.672     | 0.457  | 0.517  | 0.248     |
| rider         | 0.539     | 0.226  | 0.257  | 0.108     |
| motor         | 0.626     | 0.212  | 0.264  | 0.126     |
| bike          | 0.417     | 0.257  | 0.232  | 0.101     |

### Semantic Segmentation — SegFormer-b0 vs U-Net (BDD100K)

| Metric                   | SegFormer-b0    | U-Net (from scratch) |
|--------------------------|-----------------|----------------------|
| Test mIoU                | **0.5719**      | 0.4911               |
| Test Mean Dice           | **0.6427**      | 0.5714               |
| Inference speed (ms/img) | 8.2             | **1.9**              |
| Training time            | 59.5 min        | 401.1 min            |
| Best Val mIoU (epoch)    | 0.5568 (ep. 16) | 0.4779 (ep. 49)      |
| GPU Reserved             | 3,718 MB        | 14,644 MB            |

### Per-Class Segmentation IoU — SegFormer vs U-Net (500 test images)

| Class         | SegFormer IoU | SegFormer Dice | U-Net IoU  | U-Net Dice | Δ IoU       |
|---------------|---------------|----------------|------------|------------|-------------|
| road          | 0.9714        | 0.9855         | 0.9640     | 0.9816     | +0.0074     |
| person        | 0.5359        | 0.6474         | 0.4391     | 0.5614     | +0.0968     |
| traffic light | 0.6943        | 0.8047         | 0.6113     | 0.7366     | +0.0830     |
| traffic sign  | 0.7024        | 0.8093         | 0.5864     | 0.7230     | +0.1161     |
| car           | 0.8840        | 0.9368         | 0.8211     | 0.8976     | +0.0629     |
| truck         | 0.2709        | 0.3549         | 0.1582     | 0.2342     | +0.1127     |
| bus           | 0.3153        | 0.3741         | 0.0806     | 0.1240     | +0.2347     |
| train         | 0.0000        | 0.0000         | 0.0000     | 0.0000     | 0.0000      |
| motor         | 0.0000        | 0.0000         | 0.0000     | 0.0000     | 0.0000      |
| bike          | 0.0355        | 0.0510         | 0.0000     | 0.0000     | +0.0355     |
| **Mean**      | **0.5719**    | **0.6427**     | **0.4911** | **0.5714** | **+0.0808** |

---

## Part 2 — RAG System Summary

### Knowledge Base
- **Source:** Motor Vehicles Ordinance (MVO) 1965 + NHMP Traffic Rules
- **Chunking:** 300-word sliding window, 50-word overlap
- **Embedding model:** `all-MiniLM-L6-v2` (384-dim, L2-normalised)
- **Index:** FAISS `IndexFlatL2`
- **Coverage:** 70+ clauses across 7 violation categories

### Dual-Agent Pipeline

| Agent | Role |
|---|---|
| Observer Agent | Filters YOLO detections (conf ≥ 0.60), classifies violation type |
| Legal Consultant Agent | Queries FAISS, builds prompt, generates Citation Report via LLaMA 3.2 |

### Documented Violation Cycles

| Cycle | Frame | Violation Type | Top-1 Similarity | Applicable Law |
|-------|-------|----------------|------------------|----------------|
| 1 | frame_001 | Red Light Running | 0.6921 | Section 3.1 (MVO 1965) |
| 2 | frame_003 | Heavy Vehicle Restriction | 0.6483 | Section 7.1 |
| 3 | frame_004 | Reckless Driving | 0.6089 | Section 6.1 |

### Retrieval Depth Experiment

| k  | Behaviour | Verdict |
|----|-----------|---------|
| 3  | Precise; may miss edge-case clauses | Too conservative |
| 5  | Balanced; correct law in top-5 | **Default** |
| 10 | Rich context; low-sim chunks add noise | Too noisy |

---

## How to Run

### Part 1 — Object Detection with YOLOv11

> Runs on **Kaggle** (GPU required)

1. Upload `urban_scene_parsing/object_detection_with_YOLOv11.ipynb` to Kaggle
2. Add dataset: search `solesensei/solesensei_bdd100k` in Kaggle datasets
3. Enable GPU: **Settings → Accelerator → T4 GPU**
4. Run all cells top to bottom
5. Outputs saved to `/kaggle/working/checkpoints/` and `/kaggle/working/results/`

### Part 1 — Semantic Segmentation with SegFormer

> Runs on **Kaggle** (GPU required)

1. Upload `urban_scene_parsing/semantic_segmentation_with_SegFormer.ipynb` to Kaggle
2. Add the same BDD100K dataset
3. Enable GPU: **Settings → Accelerator → T4 GPU**
4. Run all cells top to bottom
5. SegFormer trains in ~60 min; U-Net trains in ~401 min — plan GPU hours accordingly
6. Outputs saved to `/kaggle/working/checkpoints/` and `/kaggle/working/results/`

### Part 2 — Agentic RAG Based Legal Reasoning for Traffic Safety

> Runs **locally** on your PC (Ollama requires localhost)

**Prerequisites:**
- Python 3.10+
- Ollama installed → https://ollama.com/download

**Setup:**

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/scene-parsing-legal-reasoning.git
cd DLP_Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pull the LLM model
ollama pull llama3.2

# 4. Start Ollama (keep this terminal open)
ollama serve

# 5. Place your files
# - detection_logs.json → Part2/
# - pakistan_traffic_laws.txt → Part2/law_documents/

# 6. Launch Jupyter and open the notebook
jupyter notebook "legal_reasoning/agentic_RAG_based_legal_reasoning.ipynb"
```

Run all cells top to bottom. The notebook is self-contained and runs
Phases 0–5 sequentially without errors.

---

## Environment

| Component       | Details                              |
|-----------------|--------------------------------------|
| Python          | 3.12                                 |
| PyTorch         | 2.10.0+cu128                         |
| GPU (Part 1)    | Tesla T4, 15.6 GB VRAM               |
| GPU (Part 2)    | CPU via Ollama (localhost)            |
| LLM             | LLaMA 3.2 (llama3.2:latest, 2.0 GB) |
| Embedding model | all-MiniLM-L6-v2 (384-dim)           |
| Seed            | 42 (fixed in all frameworks)         |

---

## References

1. Yu et al. (2020). BDD100K. CVPR 2020.
2. Wang et al. (2024). YOLOv11. Ultralytics Technical Report.
3. Xie et al. (2021). SegFormer. NeurIPS 2021.
4. Ronneberger et al. (2015). U-Net. MICCAI 2015.
5. Lewis et al. (2020). RAG for Knowledge-Intensive NLP. NeurIPS 2020.
6. Government of Pakistan (1965). Motor Vehicles Ordinance (MVO) 1965.
