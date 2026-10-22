Subject: Questions before implementing "Semi-Decentralized Inference in Heterogeneous GNNs" (Nazzal et al., IEEE TVT 2024)

Hello,

I have read the paper and the authors' public repository in full and started the implementation.
Before I build the data pipeline I need your decision on the following:

1. DATASET. The paper uses NYC TLC trips (Apr-Jun 2016). The authors' code actually uses `taxi_data.h5` and
   `all_graph.h5` from the CCRNN project (266 regions, Baidu link), not raw trips. Do you have either of them?
   If not, may I use the raw TLC Yellow Taxi files for Apr-Jun 2016 from nyc.gov (PARQUET)? Do you know whether
   those files still contain pick-up/drop-off longitude/latitude?
2. NODE DEFINITION. TLC 2016 has no taxi ID. I propose: node = a trip starting in the time slot (position = pick-up,
   destination = drop-off), demand = pick-ups, supply = drop-offs. Is this acceptable?
3. FINDINGS. The authors' repo does not implement what the paper describes (taxi nodes are replicated region vectors,
   all three edge types use the same matrix, LSTM unused, Fig. 12/14 numbers are typed constants sorted before plotting,
   delay code uses random positions and hand-set constants). Also Theorem 1 has internal inconsistencies
   (processing term inside vs outside the bracket; "quadratic" claim vs a cubic sum). May I implement the paper as
   written, report these discrepancies in my report, and present my own measured results even if they differ?
4. SCOPE. Which results must be reproduced (Fig. 7-14, Table III)? Is Federated Learning (Sec. IV-F) required?
5. COMPUTE. My laptop has only Intel Iris Xe (no CUDA). May I train on Google Colab/Kaggle, or does the department provide a GPU?

Thank you.
