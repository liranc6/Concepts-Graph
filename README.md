# Con Graph Project

This project involves processing and analyzing a dataset of academic papers from arXiv, focusing on computer science concepts. The project includes several Jupyter notebooks for different tasks such as creating subsets of the dataset, building graphs, and visualizing the data.

## Project Structure

- @common_concepts.txt@: A list of common concepts used in the dataset.
- @GENERAL.ipynb@: General utilities and functions used across the project.
- @CREATING_SUBSET_FROM_ARXIV_DATASET.ipynb@: Notebook for creating a subset of the arXiv dataset based on specific criteria.
- @CREATING_GRAPH.ipynb@: Notebook for building and visualizing graphs from the dataset.

## Requirements

- Python 3.9 or later
- Jupyter Notebook
- Required Python packages:
  - @numpy@
  - @matplotlib@
  - @tqdm@
  - @torch@
  - @json@
  - @requests@
  - @beautifulsoup4@
  - @networkx@
  - @plotly@

## Setup

1. Clone the repository:
   @@@sh
   git clone <repository_url>
   cd <repository_directory>
   @@@

2. Install the required packages:
   @@@sh
   pip install -r requirements.txt
   @@@

3. Open Jupyter Notebook:
   @@@sh
   jupyter notebook
   @@@

4. Open the desired notebook (e.g., @GENERAL.ipynb@) and run the cells.

## Usage

### Creating a Subset from the ArXiv Dataset

1. Open @CREATING_SUBSET_FROM_ARXIV_DATASET.ipynb@.
2. Run the cells to process the dataset and create a subset based on specific criteria.

### Building and Visualizing Graphs

1. Open @CREATING_GRAPH.ipynb@.
2. Run the cells to build a graph from the dataset and visualize it using Plotly.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the @LICENSE@ file for details.


## Report: Graph Creation and Visualization Notebook

### Overview
This notebook is designed to create and visualize a graph based on metadata from a dataset of academic papers. The graph represents relationships between different concepts (keywords) extracted from the papers. The notebook includes various functions for building the graph, calculating node attributes, and visualizing the graph using Plotly.

### Key Sections and Functions

#### Imports and Environment Setup
- The notebook begins by importing necessary libraries such as os, numpy, matplotlib, torch, networkx, and plotly.
- It sets up the environment for CUDA devices and appends the project path to the system path.

#### Metadata Extraction
- `get_metadata(data_path)`: A generator function that reads metadata from a JSON file line by line and yields it as a dictionary.

#### Graph Building
- `build_graph2(dataset)`: This function constructs a multi-graph where nodes represent concepts and edges represent co-occurrences of concepts in the same paper. Node attributes include the number of papers, minimum publish date, and hierarchy score.
- `build_graph(dataset)`: An alternative function to build a simpler graph structure with nodes and edges.

#### Graph Visualization
- `viz_graph(graph, score_range=None)`: Visualizes the graph using Plotly. It filters nodes based on a score range and creates an interactive graph with nodes and edges.
- `viz_graph1(g)`, `viz_graph2(g)`, and `draw_graph(graph)`: Additional visualization functions with slight variations in implementation.

#### Node and Edge Analysis
- `get_nodes_and_scores(G)`: Retrieves nodes and their hierarchy scores, sorted in descending order.
- `unsimilar_neighbors(connections1, connections2)`, `similar_neighbors(connections1, connections2)`, and `get_pairs(graph, similar=True)`: Functions to analyze the similarity and dissimilarity of neighboring nodes.

#### Saving Results
- `save_corpus(corpus_path, corpus)`: Saves the corpus of concepts to a text file.
- `save_common_concepts(common_concepts_path, common_concepts)`: Saves common concepts to a text file.

### Results

#### Graph Construction
- The graph is successfully built with nodes representing concepts and edges representing co-occurrences in papers. Each node has attributes such as the number of papers, minimum publish date, and hierarchy score.

#### Node Analysis
- The top 20 nodes by hierarchy score are listed, with "machine learning" having the highest score.
- Functions to find similar and dissimilar neighboring nodes are implemented but not fully executed in the notebook.

#### Visualization
- The graph is visualized using Plotly, showing an interactive representation of nodes and edges. Nodes are colored based on the number of connections.

#### Saving Data
- The corpus of concepts and common concepts are saved to text files for further analysis.

### Conclusion
The notebook provides a comprehensive approach to building and visualizing a graph from academic paper metadata. It includes functions for extracting metadata, constructing the graph, analyzing nodes, and visualizing the results. The interactive visualizations and saved data files facilitate further exploration and analysis of the graph.