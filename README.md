# DenseViT: Vision Transformer with Dense Contrastive Learning
## Introduction to Dense Contrastive Learning

Dense contrastive learning extends traditional contrastive learning from image-level to pixel-level representation learning. Unlike conventional approaches that learn global image representations, dense contrastive learning focuses on learning rich, spatially-aware features at every spatial location in the feature map.

## Proposed Architecture

```mermaid
graph TD
    A[Input Image<br/>224×224×3] --> B[Patch Embedding<br/>16×16 patches]
    B --> C[Vision Transformer Backbone<br/>Embed Dim: 192/384/768<br/>Depth: 1/12<br/>Heads: 3/6/12]
    
    C --> D[CLS Token<br/>Global Features]
    C --> E[Patch Tokens<br/>196 spatial features]
    
    D --> F[Linear Classification Head<br/>embed_dim → num_classes]
    F --> G[Classification Output<br/>Class Predictions]
    
    E --> H[Dense Projection Head<br/>MLP: embed_dim → 2048 → dense_dim]
    H --> I[Dense Features<br/>14×14×dense_dim]
    
    I --> J[Reshape to Spatial Grid<br/>B×H×W×D]
    J --> K[Dense Contrastive Learning]
    
    subgraph "Contrastive Learning Pipeline"
        K --> L[View 1 Features<br/>B×H×W×D]
        K --> M[View 2 Features<br/>B×H×W×D]
        
        L --> N[Correspondence Mining<br/>Cosine Similarity]
        M --> N
        
        N --> O[Positive Pairs<br/>Matched spatial locations]
        
        P[Memory Queue<br/>65K negative samples] --> Q[Negative Sampling]
        
        O --> R[InfoNCE Loss<br/>Per spatial location]
        Q --> R
        
        R --> S[Dense Contrastive Loss]
    end
    
    G --> T[Final Output<br/>Classification + Dense Features]
    S --> T
    
    style A fill:#e1f5fe
    style C fill:#f3e5f5
    style D fill:#e8f5e8
    style E fill:#fff3e0
    style I fill:#ffebee
    style K fill:#f1f8e9
    style S fill:#fce4ec
```

Key Idea: While the CLS token handles global classification, patch tokens are processed through a dedicated dense projection head to generate spatially-aligned contrastive features. This dual-path design enables simultaneous learning of both global semantic understanding and fine-grained spatial representations.

## Dense Contrastive Loss

The DenseContrastiveLoss class implements pixel-level contrastive learning as described in the DenseCL paper. Unlike traditional contrastive learning that operates on global image representations, this approach learns contrastive features at every spatial location in the feature map, enabling fine-grained visual understanding.Mathematical Formulation

Given two augmented views of an image, the loss is computed similar to the InfoNCE loss:

L_dense = -log(exp(sim(q_i, k_i^+) / τ) / Σⱼ exp(sim(q_i, k_j) / τ))

Where:

    q_i: Query feature at spatial location i
    k_i^+: Positive key feature at corresponding location i
    k_j: Negative key features from memory queue
    τ: Temperature parameter (typically 0.1-0.2)
    sim(·,·): Cosine similarity function


### Key Components

#### Memory Queue Mechanism
The loss function maintains a persistent memory queue to store negative samples across training iterations:
Queue Properties:

1. Persistent Storage: Survives across forward passes, accumulating diverse negative samples.
2. Normalized Features: All features in queue are L2-normalized for consistent cosine similarity computation
3. Circular Buffer: Implements wrap-around logic to efficiently manage memory

####  Correspondence Extraction
The extract_correspondence method establishes pixel-level correspondences between two augmented views of the same image.
Mathematical Foundation:
For each spatial location i in view 1, the correspondence is computed as:

```correspondence[i] = argmax_j (cosine_similarity(f1[i], f2[j]))```

Where j iterates over all spatial locations in view 2.

Key Insights:

1. Robust Matching: Uses backbone features (before projection) for correspondence to ensure semantic consistency
2. Spatial Flexibility: Allows non-rigid correspondences, accommodating augmentation-induced spatial transformations
3. Efficiency: Vectorized computation using batch matrix multiplication

### Contrastive Loss Computation Step-by-Step Process:

Step 1: Correspondence Mining
Step 2: Feature Preparation
Step 3: Positive Pair Formation
Step 4: Negative Sampling
Step 5: InfoNCE Loss Application
Step 6: Queue Update

### Key Design Decisions

1. Backbone vs Projected Features: Uses backbone features for correspondence (semantic consistency) and projected features for contrastive loss (representation learning)
2. Queue Management Strategy: Implements circular buffer with wrap-around logic for efficient memory utilization