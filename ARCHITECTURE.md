# System Architecture: Multimodal RAG for Video QA

## Overview

This document describes the architecture of the multimodal RAG pipeline for video question answering on the TVQA-Long dataset. The system evolved iteratively: each improvement was motivated by error analysis of the previous stage's failures.

---

## The Pipeline at a Glance

```
Input: Question + 5 answer options + show_name + timestamp

                    [Stage 1: RETRIEVAL]
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
                    [Stage 2: RERANKING]
                         |
              Cross-encoder (MiniLM-L-6)
              Scores each (question, clip) pair
              Promotes best match to rank 1
                         |
                      top-5 reranked
                         |
                    [Stage 3: ANSWER SELECTION]
                         |
         +----- Is question visual? -----+
         |                               |
         No (dialogue)                   Yes (visual)
         |                               |
    Cross-encoder                   CLIP (ViT-L/14)
    scores (evidence+Q,             frame extraction + 
     answer_i) for i=0..4           text-image similarity
         |                               |
         +--- predicted answer ----------+
                         |
                    [Stage 4: CONFIDENCE]
                         |
              Score margin (top-1 vs top-2)
              Faithfulness check
              Selective prediction threshold
                         |
                      Final answer + confidence
```

---

## Why This Architecture: The Reasoning Chain

### Step 1: Establish baseline, identify bottleneck

**Decision:** Start with BM25 -- the simplest effective retrieval method.

**Result (NB03):** R@1=14.7%, R@20=33.7%. Only 33.7% of gold clips appear in top-20.

**Diagnosis (NB07):** Error attribution shows 72.7% of failures are retrieval misses. The system cannot answer correctly if it never retrieves the right evidence.

**Conclusion:** Retrieval is the primary bottleneck. Improving answer selection without fixing retrieval gives diminishing returns.

### Step 2: Cheap retrieval improvements

**Decision:** Try metadata filtering and query enrichment before adding models.

**Reasoning:**
- Show-scoping (Strategy A) is free -- we always know which show a question belongs to. Restricting search from 21,793 to ~3,000 clips eliminates cross-show false positives.
- Query expansion (Strategy B) exploits the MC format -- answer options contain vocabulary that appears in gold clips but not in the question.

**Result (NB11):** Show-scoped +0.5pp, query expansion +1.2pp, combined +2.3pp at R@20.

**Conclusion:** Modest gains. The vocabulary gap is only part of the problem.

### Step 3: Dense retrieval for semantic matching

**Decision:** Add sentence-transformer embeddings (e5-small-v2) for paraphrase handling.

**Reasoning:** BM25 fails when question says "frustrated" but subtitle says "done talking to you." Dense retrieval encodes meaning, not words.

**Result (NB11):** Dense alone +3.3pp R@20. But the key finding: BM25 and dense make complementary errors.

**Decision:** Hybrid RRF fusion combines both ranked lists.

**Result (NB11):** Hybrid +11.9pp R@20 (33.7% to 45.6%). The biggest single retrieval improvement.

**Conclusion:** No single method dominates. Sparse catches rare terms, dense catches semantics. Fusion is the standard production approach (Elasticsearch 8.x, Vespa).

### Step 4: Cross-encoder reranking for precision

**Decision:** Add a cross-encoder (ms-marco-MiniLM-L-6-v2) to rescore the top-50 candidates.

**Reasoning:** Hybrid gets the right clip into top-50 (45.6%) but not necessarily top-1. A cross-encoder jointly encodes query and document, enabling it to understand relevance at a level independent encoders cannot match.

**Key architectural insight:** Cross-encoders are too slow to score all 21,793 clips (~3.5 min/query). But scoring 50 candidates takes ~500ms -- acceptable for batch and near-real-time use. This is the standard two-stage architecture: cheap recall followed by expensive precision.

**Result (NB13):** R@1 nearly doubled (12.0% to 23.6%). Universal improvement across all shows.

**Conclusion:** Cross-encoder reranking provides the precision that retrieval recall alone cannot.

### Step 5: Cross-encoder answer scoring

**Decision:** Replace token-overlap answer selection with cross-encoder scoring.

**Reasoning:** Token overlap selects the answer that shares the most words with the evidence. It cannot handle paraphrase ("picks up" != "lifts"), negation, or pragmatic inference.

**Result (NB12):** Oracle accuracy +4pp (48.4% to 52.4%). With noisy evidence, the gap narrows because token overlap is accidentally robust to irrelevant documents.

**Key finding:** Cross-encoder confidence margins are highly predictive of correctness. Selective prediction at 11% coverage achieves 86% accuracy -- valuable for production systems that can defer uncertain answers.

### Step 6: Visual evidence for visual questions

**Decision:** Question classification reveals 39.8% of questions require visual information. Add CLIP-based frame evidence.

**Reasoning:** Questions about what characters are wearing, holding, or physically doing cannot be answered from dialogue alone. No amount of text retrieval improvement helps.

**Implementation:** Extract keyframes from the timestamp range using ffmpeg (1 fps), encode with CLIP ViT-L/14, compute cosine similarity between CLIP text embeddings of each answer option and the frame embeddings.

**Result (NB15):** Visual questions improve from 29.2% (text-only) to 45.8% (combined), a +16.6pp lift. Adaptive fusion (different alpha for visual vs dialogue questions) achieves 61.2% overall on Castle subset.

### Step 7: Negative results inform the architecture

**Speaker-aware boosting (NB14):** Tested boosting clips where mentioned characters speak. Result: -0.5% at R@5.
- Why it failed: 97% of questions mention characters, and character names are already strong BM25 signals. The boost is redundant.

**Temporal context expansion (NB14):** Tested including adjacent clips in the candidate pool. Result: +0%.
- Why it failed: Adjacent clips scored by the same BM25 weights get pushed back below the original candidates.

**Token-overlap reranking (NB04):** Tested reranking BM25 candidates by token overlap with query+answers. Result: -3.4pp to -12.6pp.
- Why it failed: Token overlap destroys BM25's TF-IDF term-importance weighting by treating all tokens equally.

These negative results confirm that structural heuristics on top of BM25 are insufficient -- the improvement must come from models that understand semantics (dense retrieval, cross-encoders).

---

## Component Details

### BM25 Retrieval

```
Corpus: 21,793 subtitle clips
Tokenization: text.lower().split()
Index: BM25Okapi (rank_bm25 library)
Query: question + all 5 answer options (expanded)
Scope: restricted to same show's clips only
Output: top-50 candidates with BM25 scores
```

### Dense Retrieval

```
Model: intfloat/e5-small-v2 (384-dim embeddings)
Corpus: all 21,793 clips encoded (batch_size=256)
Index: FAISS IndexFlatIP (exact inner product after L2 normalization)
Query: question text encoded (with "query: " prefix for e5)
Output: top-50 candidates with cosine similarity scores
```

### Reciprocal Rank Fusion

```
Input: BM25 top-50 + Dense top-50
Formula: RRF_score(d) = sum(1 / (k + rank_in_list)) for k=60
Output: merged top-50 ordered by RRF score
Properties: rank-based (immune to score distribution differences),
            parameter-free (k=60 is standard), no learning required
```

### Cross-Encoder Reranker

```
Model: cross-encoder/ms-marco-MiniLM-L-6-v2
Input: list of (question, candidate_text) pairs
Processing: forward pass through 6-layer transformer with cross-attention
Output: relevance score per pair (higher = more relevant)
Speed: ~10ms per pair; 50 pairs = ~500ms per question
```

### Cross-Encoder Answer Scorer

```
Model: same cross-encoder (ms-marco-MiniLM-L-6-v2)
Input: for each answer option i: (evidence_text + " " + question, answer_i)
Processing: scores relevance of each answer to the evidence+question context
Output: 5 scores; predicted answer = argmax
Confidence: margin between top-1 and top-2 scores
```

### CLIP Visual Pipeline

```
Model: OpenAI CLIP ViT-L/14 (768-dim embeddings, loaded on MPS)
Frame extraction: ffmpeg -ss {start} -to {end} -i video.mp4 -vf "fps=1" frames/
Frame encoding: CLIP image encoder on extracted frames (batch processing)
Answer encoding: CLIP text encoder on each answer option
Score: max cosine similarity between answer text embedding and all frame embeddings
Fusion: combined_score = alpha * text_score + (1-alpha) * visual_score
Alpha: 0.3 for visual questions (more visual weight), 0.7 for dialogue (more text weight)
```

### Question Classification

```
Heuristic-based (no model needed):
1. Visual keywords in question: wearing, holding, looking, doing, gesture,
   facial, pointing, sees, watching, shown, visible, carrying, standing,
   sitting, lying, walking, running, color
2. Visual keywords in correct answer: same list
3. Grounding check: zero content-word overlap between correct answer and
   gold subtitle (after removing stopwords)

Classification: VISUAL if (1) OR (2 with 2+ matches) OR (3)
Result: 60.2% dialogue, 39.8% visual
```

---

## Data Flow

```
TVQA-Long Dataset
    |
    +-- tvqa_val_edited.json -----------> 15,253 questions (flattened from nested JSON)
    |     Keys: q, a0-a4, answer_idx,     Organized by: show -> season -> episode
    |           qid, show_name, ts,
    |           vid_name
    |
    +-- tvqa_preprocessed_subtitles.json -> 21,793 subtitle clips
    |     Keys: vid_name, sub[]            Each sub entry: {text, start, end}
    |                                      Text includes speaker labels: " House : ..."
    |
    +-- mp4_videos/{show}/season_X/ -----> 924 full episodes
          episode_Y.mp4                    Used for frame extraction in visual pipeline
```

**vid_name mapping:**
- Castle: `castle_s04e17_seg02_clip_10` -> season_4/episode_17.mp4, timestamp from `ts` field
- BBT: `s03e02_seg02_clip_10` -> season_3/episode_2.mp4 (no show prefix)
- Friends: `friends_s05e02_seg01_clip_00` -> season_5/episode_2.mp4
- House: `house_s02e05_seg02_clip_11` -> season_2/episode_5.mp4
- Grey's: `grey_s02e10_seg02_clip_05` -> season_2/episode_10.mp4
- HIMYM: `met_s03e01_seg02_clip_03` -> season_3/episode_1.mp4

---

## Data Acquisition

### Source

The dataset is hosted on HuggingFace: [Vision-CAIR/TVQA-Long](https://huggingface.co/datasets/Vision-CAIR/TVQA-Long/tree/main).

### What You Need

| Component | Size | Required? | Contents |
|-----------|------|-----------|----------|
| Annotations | ~75 MB | Yes | Question JSON + preprocessed subtitles |
| Subtitles (SRT) | ~26 MB | Yes | Full episode subtitle files (1,465 .srt) |
| Video episodes | 52.5 GB | Only for visual pipeline (NB15, NB16) | 924 full episodes across 6 shows |

### Step 1: Environment Setup

```bash
cd multimodal-rag-video-qa

# Create virtual environment
uv venv --python 3.13 .venv

# Install dependencies
uv pip install pandas numpy matplotlib seaborn scikit-learn \
    rank-bm25 nltk sentence-transformers faiss-cpu torch \
    open-clip-torch pillow jupyter httpx tqdm
```

### Step 2: Download Annotations and Subtitles

```bash
.venv/bin/python download_tvqa.py
```

This downloads:
- `data/tvqa/annotations/tvqa_val_edited.json` (15,253 questions)
- `data/tvqa/annotations/tvqa_preprocessed_subtitles.json` (21,793 subtitle clips)
- `data/tvqa/subtitles/tvqa_subtitles.zip` (1,465 episode SRT files)

These files are sufficient to run notebooks 00-14 (all text-based retrieval and answer scoring).

### Step 3: Download Video Episodes (Optional)

Required only for the visual pipeline (CLIP frame extraction in NB15 and NB16).

```bash
# Download all 11 archive parts (52.5 GB total, ~5 GB each)
.venv/bin/python download_tvqa.py --videos

# Or download individual parts (aa through ak)
.venv/bin/python download_tvqa.py --video-part aa
.venv/bin/python download_tvqa.py --video-part ab
# ... through --video-part ak
```

### Step 4: Extract Video Archives

The video files are distributed as a split tar archive (11 parts, aa through ak). Only the concatenated archive is valid -- individual parts cannot be extracted independently.

```bash
cd data/tvqa/videos

# Concatenate and extract (requires all 11 parts)
cat archive.tar.gz.a* | tar xzf - -C mp4_videos --strip-components=3

# Verify extraction (should show 6 show directories)
ls mp4_videos/
# Expected: bbt/ castle/ friends/ grey/ house/ met/

# Verify episode counts
find mp4_videos -name "*.mp4" | wc -l
# Expected: 924
```

**Directory structure after extraction:**
```
data/tvqa/videos/mp4_videos/
    bbt/         220 episodes (Big Bang Theory)
    castle/      173 episodes
    friends/     226 episodes
    grey/         55 episodes (Grey's Anatomy)
    house/       176 episodes (House M.D.)
    met/          74 episodes (How I Met Your Mother)
```

### SSL/Proxy Issues

If you are behind a corporate proxy that intercepts HTTPS (manifesting as "self-signed certificate in chain" errors), the included `download_tvqa.py` script handles this by using `httpx` with SSL verification disabled. Standard tools like `huggingface-cli` and `wget` will fail in this environment without additional configuration.

If you prefer to download manually, the HuggingFace repository structure is:
```
Vision-CAIR/TVQA-Long/
    tvqa-long-annotations/
        tvqa_preprocessed_subtitles.json
        tvqa_val_edited.json
    tvqa_subtitles.zip
    tvqa-long-videos/
        archive.tar.gz.aa
        archive.tar.gz.ab
        ... (through .ak)
```

### Disk Space Requirements

| Stage | Space Needed |
|-------|-------------|
| Annotations only (NB00-14) | ~120 MB |
| + Video archives (downloaded) | +52.5 GB |
| + Extracted episodes | +52.5 GB |
| + Extracted frames (NB15) | +1-5 GB (depends on subset) |
| **Total (full pipeline)** | **~110 GB** |

After verifying extraction, the archive parts (`archive.tar.gz.a*`) can be deleted to reclaim 52.5 GB -- but keep them if disk space permits, as re-downloading is slow.

---

## Performance Characteristics

### Latency (per question, M4 Max)

| Stage | Time | Notes |
|-------|------|-------|
| BM25 retrieval | ~1ms | Inverted index lookup |
| Dense encoding (query) | ~5ms | e5-small-v2, single vector |
| FAISS search | ~2ms | Exact inner product, 21,793 vectors |
| RRF fusion | <1ms | Rank-based merge |
| Cross-encoder reranking (50 pairs) | ~500ms | Batched forward passes |
| Cross-encoder answer scoring (5 pairs) | ~50ms | 5 pairs only |
| Frame extraction (ffmpeg) | ~200ms | Depends on clip duration |
| CLIP encoding (5-15 frames) | ~100ms | MPS backend |
| **Total (text-only pipeline)** | **~560ms** | |
| **Total (with visual)** | **~860ms** | |

### Memory (peak, all models loaded)

| Component | RAM |
|-----------|-----|
| BM25 index (21,793 docs) | ~200 MB |
| e5-small-v2 model | ~130 MB |
| FAISS index (21,793 x 384) | ~32 MB |
| Cross-encoder (MiniLM-L-6) | ~90 MB |
| CLIP ViT-L/14 | ~900 MB |
| Python overhead + data | ~500 MB |
| **Total** | **~1.9 GB** |

Well within the 64 GB available. All models can be co-resident without memory pressure.

---

## Design Decisions and Tradeoffs

| Decision | Alternatives Considered | Why This Choice |
|----------|------------------------|-----------------|
| e5-small-v2 for dense | all-MiniLM-L6-v2, bge-large | Good quality/speed balance; instruction-aware encoding |
| BM25Okapi over TF-IDF | scikit-learn TfidfVectorizer | BM25 handles document length normalization better for variable-length clips |
| FAISS IndexFlatIP over HNSW | Approximate search would be faster | 21,793 vectors is small enough for exact search; no approximation error |
| Cross-encoder for reranking over ColBERT | ColBERT token-level matching | Cross-encoder is simpler, single-score output, proven on MS MARCO |
| CLIP ViT-L/14 over SigLIP | SigLIP may be better for some tasks | CLIP is well-tested, open_clip provides easy loading, MPS support verified |
| Clip-level chunking over sliding window | Window might capture context | Questions are explicitly aligned to clip boundaries (vid_name = retrieval unit) |
| RRF over learned fusion | Trained fusion weights could optimize | RRF is parameter-free, robust, and works well without training data |
| Show-scoped retrieval always on | Could search globally | Free precision gain; no downside since show_name is always available |

---

## Notebook Dependency Graph

```
00_architecture_overview (no deps, pure design)
01_data_loading_exploration (reads annotations + subtitles)
02_text_chunking_strategies (uses subtitles from 01)
    |
    v
03_sparse_retrieval (builds BM25 index)
04_reranker (uses BM25 from 03)
05_answer_generation (uses BM25 from 03)
06_hallucination_detection (uses pipeline from 03+05)
07_end_to_end_evaluation (uses full pipeline 03+04+05)
08_observability (instruments pipeline from 07)
09_show_specific_analysis (uses pipeline, analyzes per-show)
    |
    v
10_question_classification (introduces dialogue/visual split)
11_improved_retrieval (adds dense + hybrid, uses classification from 10)
12_cross_encoder_answer (uses hybrid retrieval from 11)
13_cross_encoder_reranker (uses BM25 candidates from 03/11)
14_speaker_temporal_context (heuristic experiments, uses BM25 from 03)
15_visual_pipeline (uses video files + CLIP)
    |
    v
16_best_pipeline (combines all: 11 + 13 + 12 + 15)
```

Notebooks 03-09 can largely be run independently (each rebuilds its own BM25 index).
Notebooks 11-16 depend on models and strategies developed in prior notebooks.
