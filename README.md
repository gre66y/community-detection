**Community Detection Analysis**

This repository contains Python scripts and output files developed for a community detection research project. The goal is to analyze structural patterns in complex networks and compare the performance of different detection approaches.

**Overview**

The project includes two primary scripts:

1. sna.py

Implements community detection using classical Social Network Analysis (SNA) techniques.
It computes metrics such as centrality, modularity, clustering coefficients, and density to identify cohesive groups.
Output: ACFI_results — a structured summary of detected communities and key network metrics.

2. Comprehensive_Results_SSGA.py

Extends the analysis using additional or hybrid methods to achieve deeper and more comprehensive insights.
Output: Comprehensive_results — combined metrics and comparative analysis across methods.

Together, these scripts form a modular pipeline for network preprocessing, community detection, metric calculation, and results export.

**Features:**

Classical and hybrid community detection approaches

Graph-based metric computation

Identification of influential and boundary nodes

Comparative evaluation of algorithms

Clean, reproducible output files

**Requirements:**

Install the required Python libraries:

pip install networkx pandas numpy matplotlib


Additional libraries may be needed depending on dataset or analysis extensions.

**Usage:**

Run the SNA-based script:

python sna.py


Run the extended analysis:

python Comprehensive_Results_SSGA.py


Ensure your dataset is placed in the expected directory or modify the script paths accordingly.

**Applications:**
Social network analysis

Communication and interaction networks

Citation graphs

Any graph structure requiring community detection insights

**License:**

This project is released for academic and research use. Update licensing terms if needed.
