This repo summarizes all the resources for knowledge graph models.

<!-- [中文](./languages/chinese/) -->

## Papers
### Survey
- [Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book.pdf) (2020)
- [Representation Learning for Dynamic Graphs: A Survey](https://arxiv.org/pdf/1905.11485.pdf) (2020)
- [A Survey on Knowledge Graphs: Representation, Acquisition and Applications](https://arxiv.org/pdf/2002.00388.pdf) (2020)
- [Temporal Link Prediction: A Survey](https://link.springer.com/article/10.1007/s00354-019-00065-z) (2019)
- [A Review of Relational Machine Learning for Knowledge Graphs](https://arxiv.org/pdf/1503.00759.pdf) (2015)




### Embedding Algorithm
#### Graph Embedding
- Random Walk: [DeepWalk: online learning of social representations](https://arxiv.org/pdf/1403.6652.pdf) (KDD 2014)
- Line: [LINE: Large-scale Information Network Embedding](https://arxiv.org/pdf/1503.03578.pdf) (WWW 2015)
- node2vec: [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/pdf/1607.00653.pdf) (KDD 2016)

#### Static Knowledge Graph Embedding
<!-- More Methods: Tucker, Structured Embedding, Distance, Bilinear Model, Single Layer Model, CP, ER-MLP-->
- RESCAL: [A Three-Way Model for Collective Learning on Multi-Relational Data](https://icml.cc/2011/papers/438_icmlpaper.pdf) (ICML 2011)
- NTN [Reasoning with neural tensor networks for knowledge base completion](https://link.springer.com/content/pdf/10.1007/978-3-540-76298-0_52.pdf) (NIPS 2013)
- TransE: [Translating Embeddings for Modeling
Multi-relational Data](https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf) (NIPS 2013)
- TransH: [Knowledge Graph Embedding by Translating on Hyperplanes](https://ojs.aaai.org/index.php/AAAI/article/download/8870/8729) (AAAI 2014)
- TransX
- TransD [Knowledge Graph Embedding via Dynamic Mapping Matrix](https://aclanthology.org/P15-1067.pdf) (ACL 2015)
- TransR [Learning Entity and Relation Embeddings for Knowledge Graph](https://ojs.aaai.org/index.php/AAAI/article/download/9491/9350) (AAAI 2015)
- DistMult: [Embedding Entities and Relations for Learning and Inference in Knowledge Bases](https://arxiv.org/pdf/1412.6575.pdf) (ICRL 2015)
- HolE: [Holographic embeddings of knowledge graphs](https://arxiv.org/pdf/1510.04935.pdf) (AAAI 2016)
- ComplEx: [Complex Embeddings for Simple Link Prediction](http://proceedings.mlr.press/v48/trouillon16.pdf) (ICML 2016)
- ConvE: [Convolutional 2D Knowledge Graph Embeddings](https://arxiv.org/pdf/1707.01476.pdf) (AAAI 2018) ([Github](https://github.com/TimDettmers/ConvE))
- SimplE: [SimplE Embedding for Link Prediction in Knowledge Graphs](https://arxiv.org/pdf/1802.04868.pdf) (AAAI 2018)
- RotatE: [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://arxiv.org/pdf/1902.10197.pdf) (ICLR 2019)


Model | Score Function $\phi(h,r,t)$
:---: |:---: 
Unstructred | $-\Vert h-t \Vert_{2}^{2} $ |
Structred Embedding (SE) | $-\Vert M_{rh}h - M_{rt}t \Vert_{1} $ |
RESCAL / Binlinear | $h^{\top}R_{r}t$ |
Tucker | $\left \langle M,h \otimes r \otimes t \right \rangle $, $M \in \mathbb{R}^{3}  $
NTN | $u_{r}^{\top}$ tanh $(h^{\top}\widehat{W_{r}}t + M_{r} \begin{bmatrix} h \\ t \end{bmatrix} +b_{r})$
TransE | $-\Vert h + r - t \Vert$ |
TransH | $-\Vert (h-w_{r}^{\top}hw_{r}) + r - (t-w_{r}^{\top}tw_{r})\Vert$ |
TransX | $-\Vert g_{1,r}(h) + r - g_{2,r}(t) \Vert$ |
TransR | $-\Vert M_{r}h + r-M_{r}t \Vert$ |
DistMult | $ \left \langle h,r,t \right \rangle  $ |
HolE | $r^{\top}(Fourier^{-1}(\overline{Fourier(h)} \odot  Fourier(t)))$
ComplEx | Re $(< h,r,\overline{t}>)$ |
ConvE | $\sigma ($ vec $(\sigma([M_h;M_r]\ast \omega ))W)t$
SimplE | $-\frac{1}{2} ( <h, r, t> + <t, r^{-1}, h>)$ |
RotatE | $-\Vert h \circ r - t \Vert$ |



#### Dynamic (Temporal) Knowledge Graph Embedding
- TTransE / TTransH / TTransR: [Encoding Temporal Information for Time-Aware Link Prediction](https://aclanthology.org/D16-1260.pdf) (EMNLP 2016)
- HyTE: [HyTE: Hyperplane-based Temporally aware Knowledge Graph Embedding](https://aclanthology.org/D18-1225.pdf) (EMNLP 2018) ([Github](https://github.com/malllabiisc/HyTE))
- ConT: [Embedding Models for Episodic Knowledge Graphs](https://arxiv.org/pdf/1807.00228.pdf) (Journal of Web Semantics 2018)
- TA-TransE / TA-DistMult: [Learning Sequence Encoders for Temporal Knowledge Graph Completion](https://arxiv.org/pdf/1809.03202.pdf) (EMNLP 2018)

- DE-TransE/ DE-DistMult / DE-SimplE: [Diachronic Embedding for Temporal Knowledge Graph Completion](https://arxiv.org/pdf/1907.03143.pdf) (AAAI 2020) ([Github](https://github.com/BorealisAI/DE-SimplE))

Model | Score Function $\phi(h,r,t,\tau)$
:---: |:---: 
TTransE | $\Vert h + r - t - \tau \Vert$ |
HyTE | $-\Vert (h-w_{\tau}^{\top}hw_{\tau}) + (r-w_{\tau}^{\top}rw_{\tau}) - (t-w_{\tau}^{\top}tw_{\tau})\Vert$ |
ConT | $\left \langle M_t,h \otimes r \otimes t \right \rangle $, $M_t \in \mathbb{R}^{3}  $
TA-TransE / TA-DistMult | Uses LSTM to generate embedding $t$. Score function is TransE / DistMult.
Diachronic Embedding | Uses diachronic embeddings. Score function is TransE / DistMult / SimplE.

### Dynamic Graph Model
- Know-Evolve: [Know-Evolve: Deep Temporal Reasoning for Dynamic Knowledge Graphs](https://arxiv.org/pdf/1705.05742.pdf) (ICML 2017)
- CTDNE: [Continuous-Time Dynamic Network Embeddings](https://dl.acm.org/doi/pdf/10.1145/3184558.3191526) (WWW 2018)
- DyRep: [DyRep: Learning Representations over Dynamic Graphs](https://openreview.net/pdf?id=HyePrhR5KX) (ICLR 2019)
- RE-NET: [Recurrent Event Network: Autoregressive Structure Inference over Temporal Knowledge Graphs](https://arxiv.org/pdf/1904.05530.pdf) (EMNLP 2020) ([Github](https://github.com/INK-USC/RE-Net))
- TGAT: [Inductive Representation Learning on Temporal Graphs](https://arxiv.org/pdf/2002.07962.pdf) (ICLR 2020)
- CygNet: [Learning from History: Modeling Temporal Knowledge Graphs with Sequential Copy-Generation Networks](https://arxiv.org/pdf/2012.08492.pdf) (AAAI 2021)


- DynamicTriad: [Dynamic Network Embedding by Modeling Triadic Closure Process](https://ojs.aaai.org/index.php/AAAI/article/view/11257/11116) (AAAI 2018) 
- NetWalk: [NetWalk: A Flexible Deep Embedding Approach for Anomaly Detection in Dynamic Networks]() (KDD 2018)
- WD-GCN / CD-GCN: [Dynamic Graph Convolutional Networks](https://arxiv.org/pdf/1704.06199.pdf) (Pattern Recognition 2019)
- Dyngraph2vec: [dyngraph2vec: Capturing Network Dynamics using Dynamic Graph Representation Learning](https://arxiv.org/pdf/1809.02657.pdf) (Knowledge-Based Systems 2020)
- EvolveGCN: [EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs](https://arxiv.org/pdf/1902.10191.pdf) (AAAI 2020)
- DySAT: [DySAT: Deep Neural Representation Learning on Dynamic Graphs via Self-Attention Networks](http://yhwu.me/publications/dysat_wsdm20.pdf) (WSDE 2020)
- JODIE: [Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks](https://arxiv.org/pdf/1908.01207.pdf) (KDD 2019)
- MMDNE: [Temporal Network Embedding with Micro-and Macro-dynamics](https://arxiv.org/pdf/1909.04246.pdf) (CIKM 2019)
- HierTCN: [Hierarchical Temporal Convolutional Networks for Dynamic Recommender Systems](https://arxiv.org/pdf/1904.04381.pdf) (WWW 2019)

- DGNN: [Streaming Graph Neural Networks](https://arxiv.org/pdf/1810.10627.pdf) (SIGIR 2020)
- CAW: [Inductive Representation Learning in Temporal Networks via Causal Anonymous Walks](https://arxiv.org/pdf/2101.05974.pdf) (ICLR 2021)
- EvoNet: [Time-Series Event Prediction with Evolutionary State Graph](https://arxiv.org/pdf/1905.05006.pdf) (WSDM 2021)
- APAN: [APAN: Asynchronous Propagation Attention Network for Real-time Temporal Graph Embedding](https://arxiv.org/pdf/2011.11545.pdf) (SIGMOD 2021)

- DynGEM: [DynGEM: Deep Embedding Method for Dynamic Graphs](https://arxiv.org/pdf/1805.11273.pdf) (IJCAI 2017)

### Graph Model
- Graph-SAGE: [Inductive Representation Learning on Large Graphs](https://proceedings.neurips.cc/paper/2017/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf) (NIPS 2017)
- GCN: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907.pdf) (ICLR 2017)
- GCRN: [Structured Sequence Modeling with Graph Convolutional Recurrent Networks](https://arxiv.org/pdf/1612.07659.pdf) (ICONIP 2018)
- GAT: [Graph Attention Networks](https://arxiv.org/pdf/1710.10903.pdf) (ICLR 2018)
- R-GCN: [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/pdf/1703.06103.pdf) (ESWC 2018)



## Datasets
### Knowledgr Graph Dataset
- YAGO [paper](https://hal.archives-ouvertes.fr/hal-01472497/file/www2007.pdf) (WWW 2007)
- Wikidata [paper](https://link.springer.com/content/pdf/10.1007/978-3-319-11964-9_4.pdf) (2014)
- Google KG
- WordNet 
    - [WN18]() 40,943 entities, 18 relation types, 151,442 triples.
    - [WN18RR]()
- Freebase 
    - [FB15K](https://paperswithcode.com/dataset/fb15k) 14,951 entities, 1345 relation types, 592,213 triples.
    - [FB15k-401]() 14,541 entities, 401 relation types, 560,209 triples
    - [RB15k-237]() 14,541 entities, 237 relation types, 310,116 triples
- Global Database of Events, Language, and Tone (GDELT)
- Integrated Conflict Early Warning System (ICEWS)
    - This dataset described interactions between nations over years.
    - [ICEWS 2014]()
    - [ICEWS 2005-15]()
    - [ICEWS18]()

### Citation Network Dataset
- Cora 2798 nodes, 5429 edges
- Citeseer 3327 nodes, 4732 edges
- Pumbed 19,717 nodes, 44,338 edges
- DBpedia [paper](https://link.springer.com/content/pdf/10.1007/978-3-540-76298-0_52.pdf) (ISWC/ASWC 2007)
- HEP-TH
- [OGB](https://ogb.stanford.edu/)

### Other Dataset
- PPI (Protein-Protein Interaction)
## Libararies

- [Deep Graph Library (DGL)](https://www.dgl.ai/)
- [PyTorch Geometric Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal)
- [PyKEEN](https://github.com/pykeen/pykeen)
- [GraphVite](https://github.com/DeepGraphLearning/graphvite)
- [Stellar Graph](https://github.com/stellargraph/stellargraph)
- [DGL-KE](https://github.com/awslabs/dgl-ke)
- [OpenKE](https://github.com/thunlp/OpenKE)

## Others
