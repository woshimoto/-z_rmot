# Compact Tracklet-Centric RMOT Figure

```mermaid
flowchart LR
    classDef input fill:#eef4ff,stroke:#4a6ee0,stroke-width:1.1px,color:#111;
    classDef module fill:#f8f9fb,stroke:#6b7280,stroke-width:1.0px,color:#111;
    classDef highlight fill:#eefbf3,stroke:#2f9d62,stroke-width:1.1px,color:#111;
    classDef output fill:#fff6e8,stroke:#d48a1f,stroke-width:1.1px,color:#111;

    V["Video"]:::input
    Q["Referring Expression"]:::input

    P["Open-Vocabulary Proposals"]:::module
    T["Tracklet Construction"]:::module
    M["Tracklet-Centric Visual-Language Memory"]:::highlight
    D["Structured Query Decomposition"]:::module
    R["Query-Tracklet Matching"]:::module
    O["Referred Object Tracks"]:::output

    V --> P --> T --> M --> R --> O
    Q --> D --> R
```

## Suggested Caption

**Overview of the proposed tracklet-centric zero-shot RMOT framework.**  
Given a video and a referring expression, the method first generates open-vocabulary proposals and associates them into candidate tracklets. It then performs visual-language reasoning at the tracklet level by building structured tracklet memories, which are matched with decomposed query constraints to predict the referred object tracks.

