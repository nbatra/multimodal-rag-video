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

The pipeline evolved iteratively through error analysis: each improvement was motivated by diagnosing failures at the previous stage. The final unified architecture is:

```
Input: Question + 5 answer options + show_name + timestamp

                    [Stage 1: HYBRID RETRIEVAL]
                           |
         +-----------------+-----------------+
         |                                   |
    BM25 (show-scoped,                Dense (e5-small-v2)
     query-expanded)                   FAISS cosine search
         |                                   |
         +--- top-50 ----+---- top-50 -------+
                         |
               Reciprocal Rank Fusion (k=60)
                         |
                      top-50 fused
                         |
                    [Stage 2: CROSS-ENCODER RERANKING]
                         |
              ms-marco-MiniLM-L-6-v2 scores
              each (question, clip) pair jointly
                         |
                      top-5 reranked
                         |
                    [Stage 3: ANSWER SELECTION]
                         |
         +----- Is question visual? -----+
         |                               |
         No (dialogue)                   Yes (visual + video available)
         |                               |
    Cross-encoder                   CLIP ViT-L/14
    scores (evidence+Q,             frame extraction (ffmpeg 1fps)
     answer_i) for i=0..4           text-image similarity per answer
         |                               |
         +-------- adaptive fusion ------+
         |    alpha=0.7 (dialogue)       |
         |    alpha=0.3 (visual)         |
         +--- predicted answer ----------+
                         |
                    [Stage 4: CONFIDENCE]
                         |
              Score margin (top-1 vs top-2)
              Selective prediction threshold
                         |
                      Final answer + confidence
```

**Why this architecture (the reasoning chain):**
1. BM25 baseline achieved R@20=33.7% -- error analysis showed 72.7% of failures were retrieval misses
2. Dense retrieval alone: +3.3pp. But BM25 and dense make complementary errors (sparse catches rare terms, dense catches semantics)
3. Hybrid RRF: +11.9pp R@20 -- the single biggest retrieval improvement
4. Cross-encoder reranking: R@1 nearly doubled (12% to 23.6%) -- expensive precision on the cheap recall set
5. Cross-encoder answer scoring: +4pp over token overlap -- understands paraphrase and entailment
6. CLIP visual: +16.6pp on visual questions -- 39.8% of questions are unanswerable from text alone

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

The progression follows a clear diagnostic logic -- each improvement was motivated by error analysis of the previous stage:

1. **Start with BM25 baseline** -- establishes what pure lexical matching achieves (R@20=33.7%)
2. **Error analysis reveals 72.7% failures are retrieval** -- retrieval is the bottleneck, not answer selection
3. **Show-scoping + query expansion** -- cheapest fixes first, exploit metadata and MC format (+2.3pp)
4. **Dense retrieval** -- addresses the paraphrase gap (question says "frustrated", subtitle says "done talking to you")
5. **Hybrid RRF** -- BM25 and dense have complementary strengths; fusing captures both (+11.9pp)
6. **Cross-encoder reranking** -- BM25/dense get the right clip into top-50 (45.6%) but not top-1; cross-encoder provides the semantic precision to promote it (R@1 doubled)
7. **Cross-encoder answer scoring** -- token overlap for answer selection hits a ceiling at ~48% oracle; cross-encoder understands entailment and paraphrase (+4pp)
8. **Question classification** -- heuristic analysis reveals 39.8% of questions require visual information not in subtitles; text improvements have diminishing returns on these
9. **CLIP visual pipeline** -- extract frames from video episodes, encode with CLIP ViT-L/14, compute text-image similarity per answer option; adaptive fusion combines visual and text evidence (+16.6pp on visual questions)
10. **Speaker/temporal heuristics** -- tested and rejected; confirms that surface-level structural signals don't add value beyond what BM25 already captures

Each step either improved accuracy (kept) or was honestly reported as a negative result (NB04 token reranking, NB14 speaker/temporal). The architecture doc (ARCHITECTURE.md) contains the full reasoning chain with specific numbers at each stage.

## Visual Model: CLIP ViT-L/14

The visual pipeline uses OpenAI's CLIP ViT-L/14 to score answer options against extracted video frames. Understanding its capabilities and limitations is important for interpreting results.

**What CLIP can detect:**
- Scene type and setting (indoor/outdoor, office, hospital, apartment)
- Objects and props (coffee cups, whiteboards, phones, weapons)
- General actions (standing, sitting, walking, pointing, hugging)
- Colors and clothing (red shirt, blue dress, lab coat)
- Spatial relationships (person near door, object on table)

**What CLIP cannot do:**
- Identify specific characters by face (it does not know who "Sheldon" or "Beckett" is)
- Detect fine-grained emotions (subtle frustration vs. annoyance vs. disappointment)
- Reason temporally across frames (it scores each frame independently)
- Read text in scenes (whiteboard content, phone screens)
- Understand character relationships or social dynamics from visual cues alone

**How to improve in production:**
- Add a face recognition module (train on show-specific face crops) for character identification
- Use video-language models (VideoCLIP, InternVideo, Video-LLaVA) for temporal reasoning
- Fine-tune CLIP on TV show frames with character-labeled captions
- Use OCR for text-heavy scenes (whiteboards, computer screens)
- Ensemble multiple visual models for different question subtypes

## Technical Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Sparse Retrieval | rank-bm25 | BM25 keyword search |
| Dense Retrieval | sentence-transformers (e5-small-v2) | Semantic embedding + FAISS |
| Cross-Encoder | ms-marco-MiniLM-L-6-v2 | Reranking + answer scoring |
| Visual | CLIP ViT-L/14 (HuggingFace transformers) | Frame embedding + text-image similarity |
| Frame Extraction | ffmpeg | Keyframe extraction at 1 fps |
| Data Science | pandas, numpy, matplotlib, seaborn | Analysis and visualization |

## Hardware and Production Considerations

**Development environment (this project):**
- Apple M4 Max, 64 GB RAM
- MPS backend for PyTorch inference
- 644 GB available disk (53 GB used for video episodes)
- All models run locally -- no API calls required

**Design choices constrained by hardware:**

| Choice Made | Why | Production Alternative |
|-------------|-----|----------------------|
| e5-small-v2 (384-dim) | Fast encoding on CPU/MPS, fits in RAM alongside other models | e5-large-v2 or bge-large (1024-dim) on GPU for better recall |
| MiniLM-L-6-v2 (6 layers) | Fast cross-encoder inference (~1ms/pair) | 12-layer or fine-tuned cross-encoder for higher precision |
| FAISS IndexFlatIP (exact search) | 21,793 vectors is small enough for brute-force | HNSW or IVF indices for million-scale corpora |
| CLIP ViT-L/14 | Fits on MPS, well-tested, good zero-shot performance | SigLIP, InternVL, or fine-tuned CLIP for domain-specific tasks |
| Sequential per-question processing | Simple, debuggable | Batch GPU inference with async frame extraction |
| 200-question eval subset | Full 15K would take hours on single machine | Distributed evaluation across GPU cluster |

**Production scaling recommendations:**
- Use GPU clusters (A100/H100) for batch embedding of the full corpus
- Deploy cross-encoder reranking as a microservice with batched inference
- Use approximate nearest neighbor search (HNSW, ScaNN) for corpora > 100K documents
- Consider LLM-based answer scoring (GPT-4, Claude) for the final answer selection stage -- more expensive but substantially more capable at reasoning
- Pre-extract and cache video frame embeddings rather than computing on-the-fly
- Add a dedicated face recognition model for character identification in visual questions

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
