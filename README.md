# Path-MGCN

## Path-MGCN: a pathway activity-based multi-view graph convolutional network for determining spatial domains
Spatial transcriptomics (ST) comprehensively measure the gene expression profiles while preserving the spatial information. Accumulated computational frameworks have been proposed to identify spatial domains, one of the fundamental tasks of ST data analysis, to understand the tissue architecture. However, current methods often overlook pathway-level functional context and struggle with data sparsity. Therefore, we develop Path-MGCN, a multi-view graph convolutional network (GCN) with attention mechanism, which integrates pathway information. We first calculate spot-level pathway activity scores via gene set variation analysis from gene expression and construct distinct adjacency graphs representing spatial and functional proximity. A multi-view GCN learns spatial, pathway, and shared embeddings adaptively fused by attention and followed by a Zero-inflated negative binomial decoder to retain the original transcriptome information. Comprehensive evaluations across diverse datasets (human dorsolateral prefrontal cortex, breast cancer and mouse brain) at various resolution demonstrate Path-MGCN’s superior accuracy and robustness, significantly outperforming state-of-the-art methods and maintaining high performance across different pathway databases (Kyoto Encyclopedia of Genes and Genomes, Gene Ontology, Reactome). Crucially, Path-MGCN enhances biological interpretability, enabling the identification of Tertiary lymphoid structure-like regions and spatially resolved metabolic heterogeneity (hypoxia, glycolysis, AMP-activated protein kinase signaling) linked to tumor progression stages in human breast cancer. By effectively integrating functional context, Path-MGCN advances ST analysis, providing an accurate and interpretable framework to dissect tissue heterogeneity and enables detailed spatial mapping of molecular pathways that highlights potential targeted therapeutic strategies crucial for developing safe and effective synergistic anti-tumor therapies.

---

## Install
```bash
conda install -f environment.yml
```

---

## Data
The ST datasets and pathway databases are publicly available:

1. **10x Visium ST** dataset of human dorsolateral prefrontal cortex (DLPFC): [http://spatial.libd.org/spatialLIBD](http://spatial.libd.org/spatialLIBD).
2. **10x Visium ST** datasets of human breast cancer: [https://support.10xgenomics.com/spatial-gene-expression/datasets](https://support.10xgenomics.com/spatial-gene-expression/datasets).
3. **Stereo-seq** dataset of mouse olfactory bulb tissue: [https://github.com/JinmiaoChenLab/SEDR_analyses](https://github.com/JinmiaoChenLab/SEDR_analyses).
4. **Slide-seq V2** dataset of mouse olfactory bulb tissue: [https://singlecell.broadinstitute.org/single_cell/study/SCP815/highly-sensitive-spatial-transcriptomics-at-near-cellular-resolution-with-slide-seqv2#study-summary](https://singlecell.broadinstitute.org/single_cell/study/SCP815/highly-sensitive-spatial-transcriptomics-at-near-cellular-resolution-with-slide-seqv2#study-summary).
5. **STARmap** dataset of mouse visual cortex tissue: [https://figshare.com/articles/dataset/STARmap_datasets/22565209](https://figshare.com/articles/dataset/STARmap_datasets/22565209).
6. **osmFISH** dataset of mouse somatosensory cortex tissue: [http://linnarssonlab.org/osmFISH/availability](http://linnarssonlab.org/osmFISH/availability).
7. **KEGG PATHWAY**: [https://www.kegg.jp/kegg/pathway.html](https://www.kegg.jp/kegg/pathway.html).
8. **Gene Ontology (GO)**: [https://geneontology.org/docs/download-ontology](https://geneontology.org/docs/download-ontology).
9. **Reactome Pathway**: [https://reactome.org/download-data](https://reactome.org/download-data).


---

## Usage
Example on **DLPFC**:
```bash
python Path-MGCN_DLPFC.py
```

---

## Citation
Zhou, Qirui, et al. Path-MGCN: a pathway activity-based multi-view graph convolutional network for determining spatial domains. ***Briefings in Bioinformatics*** 26(4), 2025: bbaf365. https://doi.org/10.1093/bib/bbaf365