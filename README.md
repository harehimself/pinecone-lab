<p align="center">
   <img src="https://raw.githubusercontent.com/harehimself/pinecone-lab/main/pinecone-lab.png">
</p>
<p align="center">
   Experimenting with Pinecone as vector data is becoming the standard for AI-native systems. Purpose of the project is learning and experimentation, with the intent of implementing all core Pinecone DB capabilities.
</p>
<p align="center">
  <a href="https://github.com/harehimself/pinecone-lab/graphs/contributors"><img src="https://img.shields.io/github/contributors/harehimself/pinecone-lab" alt="Contributors"></a>
  <a href="https://github.com/harehimself/pinecone-lab/network/members"><img src="https://img.shields.io/github/forks/harehimself/pinecone-lab" alt="Forks"></a>
  <a href="https://github.com/harehimself/pinecone-lab/stargazers"><img src="https://img.shields.io/github/stars/harehimself/pinecone-lab" alt="Stars"></a>
  <a href="https://github.com/harehimself/pinecone-lab/issues"><img src="https://img.shields.io/github/issues/harehimself/pinecone-lab" alt="Issues"></a>
  <a href="https://github.com/harehimself/pinecone-lab/blob/main/LICENSE"><img src="https://img.shields.io/github/license/harehimself/pinecone-lab" alt="MIT License"></a>
</p>

---

## Table of Contents
  - [Pinecone Learning & Experimentation](#pinecone-learning--experimentation)
  - [Features](#features)
  - [Installation](#installation)
  - [Examples](#examples)
  - [Jupyter Notebooks](#jupyter-notebooks)
  - [Project Structure](#project-structure)
  - [Future Enhancements](#future-enhancements)
  - [License](#license)

<br>

## Pinecone Learning & Experimentation
A comprehensive learning project for exploring Pinecone vector database capabilities through hands-on examples and interactive notebooks.

<br>

## Features
- Complete Pinecone DB operations coverage
- Vector search and similarity matching
- Metadata filtering and namespaces
- Hybrid search (dense + sparse vectors)
- Interactive Jupyter notebooks
- Utility functions and configuration management

<br>

## Installation
1. Clone the repository:
```bash
git clone https://github.com/harehimself/pinecone-lab.git
cd pinecone-lab
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure API key as environment variable:
```bash
# Variable Name: PINECONE_API_KEY
# Variable Value: your-api-key-here
```

<br>

## Examples
The `src/examples` directory contains 8 standalone scripts demonstrating core features:

- **Basic Operations**: `python -m src.examples.01_basic_operations`
- **Vector Operations**: `python -m src.examples.02_vector_operations`
- **Namespaces**: `python -m src.examples.03_namespaces`
- **Metadata Filtering**: `python -m src.examples.04_metadata_filtering`
- **Sparse Vectors**: `python -m src.examples.05_sparse_vectors`
- **Hybrid Search**: `python -m src.examples.06_hybrid_search`
- **Integrated Embedding**: `python -m src.examples.07_integrated_embedding`
- **Backup & Restore**: `python -m src.examples.08_backup_restore`


<br>

## Jupyter Notebooks
Interactive notebooks for hands-on exploration:
```bash
jupyter notebook notebooks/01_quickstart.ipynb
jupyter notebook notebooks/02_advanced_usage.ipynb
```

<br>

## Project Structure
- `src/config.py`: Configuration and environment variables
- `src/utils.py`: Utility functions for Pinecone operations
- `src/examples/`: Individual example scripts
- `notebooks/`: Interactive learning notebooks

<br>


## Future Enhancements
- Real embedding models (OpenAI, SentenceTransformers)
- Real-world dataset examples
- Performance benchmarking tools
- Vector space visualization tools

<br>

## License
MIT License © 2025 [HareLabs](https://github.com/harehimself)
