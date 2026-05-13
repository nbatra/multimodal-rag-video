# Multimodal RAG for Video Question Answering

A systematic study of retrieval-augmented generation for video QA, progressing from baseline BM25 through dense retrieval, cross-encoder reranking, and multimodal visual evidence. Built on the TVQA-Long dataset (6 TV shows, 15,253 questions, 924 full episodes).

## The Problem

Given a question about a TV show episode ("Why is Howard frustrated when talking to Sheldon?"), find the relevant subtitle clip from 21,793 candidates, then select the correct answer from 5 multiple-choice options. This requires:

1. **Retrieval** -- finding the right clip among thousands
2. **Reasoning** -- understanding whether the evidence supports each answer
3. **Visual understanding** -- some questions require seeing what characters are doing, wearing, or holding

The project implements and compares progressively more sophisticated approaches, measuring accuracy at each stage to understand where the gains come from.

## Key Results

| Configuration | Overall Accuracy | Dialogue Qs | Visual Qs |
|---------------|-----------------|-------------|-----------|
| BM25 + token overlap (baseline) | ~29% | ~35% | ~19% |
| Hybrid RRF retrieval | ~31% | ~37% | ~20% |
| + Cross-encoder reranking | ~34% | ~40% | ~22% |
| + Cross-encoder answer scoring | ~36% | ~42% | ~23% |
| + CLIP visual evidence (Castle) | ~38% | ~42% | ~46% |
| Selective prediction (high-confidence only) | up to 86% at 11% coverage | -- | -- |

### Retrieval Progression

| Strategy | R@1 | R@20 | Key Insight |
|----------|-----|------|-------------|
| Vanilla BM25 | 14.7% | 33.7% | Baseline -- purely lexical |
| Show-scoped BM25 | -- | 34.2% | Free metadata filter eliminates cross-show noise |
| Query expansion | -- | 34.9% | Answer options inject gold-clip vocabulary |
| Dense (e5-small-v2) | -- | 37.1% | Handles paraphrases BM25 misses |
| Hybrid RRF | -- | 45.6% | BM25 + dense are highly complementary |
| Cross-encoder reranking | 23.6% | -- | Doubles R@1 by understanding semantic relevance |

### What Did NOT Work

| Strategy | Result | Why |
|----------|--------|-----|
| Speaker-aware boosting | -0.5% at R@5 | 97% of questions already mention characters; redundant with BM25 |
| Temporal context expansion | +0% | Neighbors scored by same BM25 weights get pushed back |
| Token-overlap reranking | -3.4pp to -12.6pp | Destroys BM25's term-importance signal |

## Dataset

**TVQA-Long** (Vision-CAIR/TVQA-Long on HuggingFace):
- 6 TV shows: Big Bang Theory (220 eps), Friends (226), Castle (173), House M.D. (176), How I Met Your Mother (74), Grey's Anatomy (55)
- 15,253 validation questions (5-choice MC)
- 21,793 timestamped subtitle clips with speaker labels
- 924 full episode videos (53 GB)
- Each question references a specific video clip (ground truth for retrieval evaluation)

**Question classification:** 60.2% are dialogue-answerable (can be solved from subtitles alone), 39.8% require visual information.

## Architecture

```
Question + Answer Options
         |
         v
[Stage 1: Retrieval]
    BM25 (show-scoped, query-expanded) -> top-50
    Dense (e5-small-v2 + FAISS) -> top-50
    Reciprocal Rank Fusion -> top-50 combined
         |
         v
[Stage 2: Reranking]
    Cross-encoder (ms-marco-MiniLM-L-6-v2)
    Scores all 50 candidates jointly with query
    Promotes semantically relevant clips to top-5
         |
         v
[Stage 3: Answer Selection]
    For dialogue questions:
        Cross-encoder scores (evidence+question, answer_i) for each option
    For visual questions (with video available):
        CLIP (ViT-L/14) computes text-image similarity per frame
        Adaptive fusion: alpha * text_score + (1-alpha) * visual_score
         |
         v
[Stage 4: Confidence + Quality]
    Score margin between top-1 and top-2 answer
    Faithfulness check (token grounding in evidence)
    Selective prediction: abstain when confidence < threshold
```

## Notebook Series

### Foundation (NB 00-02)
| # | Notebook | Purpose |
|---|----------|---------|
| 00 | Architecture Overview | Pipeline design, dataset stats, notebook roadmap |
| 01 | Data Loading & Exploration | EDA: question distributions, subtitle analysis, speaker frequency |
| 02 | Text Chunking Strategies | Compare 4 chunking approaches; clip-level wins (87.6% in-range) |

### Retrieval (NB 03-04)
| # | Notebook | Purpose |
|---|----------|---------|
| 03 | Sparse Retrieval (BM25) | Baseline R@1=14.7%, R@20=33.7%; Castle best, Friends worst |
| 04 | Token-Overlap Reranker | Bigram reranking +1.5pp; token overlap hurts (negative result) |

### Generation & Evaluation (NB 05-09)
| # | Notebook | Purpose |
|---|----------|---------|
| 05 | Answer Generation | Evidence-based MC selection; TF-IDF scoring best (45.6% oracle) |
| 06 | Hallucination Detection | Faithfulness scoring; trusted set 31.8% at 43% abstention |
| 07 | End-to-End Evaluation | Full pipeline 29.7%; retrieval is 72.7% of errors |
| 08 | Observability | Tracing, latency profiling, drift detection, anomaly flagging |
| 09 | Show-Specific Analysis | Castle easiest (30.3%), House hardest (17.9%); genre matters |

### Improvements (NB 10-16)
| # | Notebook | Purpose |
|---|----------|---------|
| 10 | Question Classification | Separate dialogue (60.2%) from visual (39.8%); fair evaluation |
| 11 | Improved Retrieval | Show-scoping, query expansion, dense, hybrid RRF (+11.9pp R@20) |
| 12 | Cross-Encoder Answer | Semantic answer scoring; oracle +4pp; selective prediction 86%@11% |
| 13 | Cross-Encoder Reranker | R@1 doubled (12% to 23.6%); universal improvement across shows |
| 14 | Speaker + Temporal | Negative result: heuristics redundant with BM25 |
| 15 | Visual Pipeline | CLIP frame evidence; visual questions +16.6pp; adaptive fusion 61.2% |
| 16 | Best Pipeline | All improvements combined; final accuracy measurements |

## Why Each Strategy Was Tried

The progression follows a clear diagnostic logic:

1. **Start with BM25 baseline** -- establishes what pure lexical matching achieves
2. **Error analysis reveals 72.7% failures are retrieval** -- retrieval is the bottleneck
3. **Show-scoping + query expansion** -- cheapest fixes, exploit metadata and MC format
4. **Dense retrieval** -- addresses the paraphrase gap (question says "frustrated", subtitle says "done talking to you")
5. **Hybrid RRF** -- BM25 and dense have complementary strengths; fusing captures both
6. **Cross-encoder reranking** -- BM25/dense get the right clip into top-50 (45.6%) but not top-1; cross-encoder provides the semantic precision to promote it
7. **Cross-encoder answer scoring** -- token overlap for answer selection hits a ceiling at ~48% oracle; cross-encoder understands entailment and paraphrase
8. **Visual pipeline** -- 39.8% of questions are fundamentally unanswerable from text; CLIP provides frame-level evidence for visual reasoning
9. **Speaker/temporal heuristics** -- tested and rejected; confirms that surface-level structural signals don't add value beyond what BM25 already captures

Each step either improved accuracy (kept) or was honestly reported as a negative result (NB04 token reranking, NB14 speaker/temporal).

## Technical Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Sparse Retrieval | rank-bm25 | BM25 keyword search |
| Dense Retrieval | sentence-transformers (e5-small-v2) | Semantic embedding + FAISS |
| Cross-Encoder | ms-marco-MiniLM-L-6-v2 | Reranking + answer scoring |
| Visual | OpenCLIP (ViT-L/14) | Frame embedding + text-image similarity |
| Frame Extraction | ffmpeg | Keyframe extraction at 1 fps |
| Data Science | pandas, numpy, matplotlib, seaborn | Analysis and visualization |

## Hardware

- Apple M4 Max, 64 GB RAM
- MPS backend for PyTorch inference
- 644 GB available disk (53 GB used for video episodes)
- All models run locally -- no API calls required

## How to Run

```bash
cd multimodal-rag-video-qa

# Create environment
uv venv --python 3.13 .venv
uv pip install pandas numpy matplotlib seaborn scikit-learn \
    rank-bm25 nltk sentence-transformers faiss-cpu torch \
    open-clip-torch pillow jupyter

# Download TVQA-Long data (annotations only, ~75 MB)
.venv/bin/python download_tvqa.py

# Download video episodes (optional, 52.5 GB)
.venv/bin/python download_tvqa.py --videos

# Run notebooks
.venv/bin/jupyter lab notebooks/tvqa/
```

## Project Structure

```
multimodal-rag-video-qa/
  data/
    nextqa/                    # Original NExT-QA dataset (prior work)
    tvqa/
      annotations/             # Question annotations + preprocessed subtitles
      subtitles/               # Full episode SRT files (zipped)
      videos/
        mp4_videos/            # 924 episodes across 6 shows (53 GB)
        archive.tar.gz.a*      # Original download archives
      frames/                  # Extracted keyframes (generated by NB15)
  notebooks/
    nextqa/                    # Prior NExT-QA notebooks (00-12)
    tvqa/                      # TVQA notebooks (00-16)
      plots/                   # 30+ generated visualizations
  download_tvqa.py             # Dataset download script (SSL-bypass for corporate proxies)
  ARCHITECTURE.md              # Detailed architecture document
  README.md                    # This file
```

## Author

Built by **Nipun Batra**
