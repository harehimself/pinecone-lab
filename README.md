<div align="center">
<a href="https://pinecone.io" target="_blank" title="Pinecone Vector Database"><img width="196px" alt="Pinecone Logo" src="https://raw.githubusercontent.com/harehimself/pinecone/refs/heads/main/pinecone-logo.png"></a>

<a name="readme-top"></a>

Pinecone Learning & Experimentation
==================

Experimenting with Pinecone as vector data is becoming the standard for AI-native systems. Purpose of the project is learning and experimentation, with the intent of implementing all core Pinecone DB capabilities.


**↘  Share the Project  ↙**\
https://github.com/harehimself/pinecone


</div>


## Setup

1. Clone this repository.
2. Create and activate a virtual environment:
- Using venv: `python -m venv venv`
- Activate on Windows: `venv\Scripts\activate`
- Activate on macOS/Linux: `source venv/bin/activate`
1. Install dependencies: `pip install -r requirements.txt`
2. Configure a Windows System Environment Variable for your API key.
- Variable Name: `PINECONE_API_KEY`
- Variable Value: `your-api-key-here`


## Examples

The `src/examples` directory contains standalone Python scripts demonstrating Pinecone features:

- 1. **Basic Operations**: Creating indexes, upserting, querying, and deleting vectors
`python -m src.examples.01_basic_operations`

- 2. **Vector Operations**: Batch operations and vector manipulations
`python -m src.examples.02_vector_operations`

- 3. **Namespaces**: Working with multiple namespaces for data isolation
`python -m src.examples.03_namespaces`

- 4. **Metadata Filtering**: Filtering query results using metadata
`python -m src.examples.04_metadata_filtering`

- 5. **Sparse Vectors**: Working with sparse vectors for keyword search
`python -m src.examples.05_sparse_vectors`

- 6. **Hybrid Search**: Combining dense and sparse vectors
`python -m src.examples.06_hybrid_search`

- 7. **Integrated Embedding**: Using Pinecone's integrated embedding capabilities
`python -m src.examples.07_integrated_embedding`

- 8. **Backup & Restore**: Backing up and restoring indexes
`python -m src.examples.08_backup_restore`


## Jupyter Notebooks

The `notebooks` directory contains interactive notebooks for exploring Pinecone:

1. **Quickstart**: Basic operations and concepts
`jupyter notebook notebooks/01_quickstart.ipynb`

2. **Advanced Usage**: More complex features and use cases
`jupyter notebook notebooks/02_advanced_usage.ipynb`


## Project Structure

- `src/config.py`: Configuration and environment variables
- `src/utils.py`: Utility functions for working with Pinecone
- `src/examples/`: Individual example scripts
- `notebooks/`: Jupyter notebooks for interactive learning


## Notes

- Each example creates its own index with a unique name
- Examples have a `clean_up_index()` function that is commented out by default
- Uncomment the cleanup code to delete indexes after running examples
- For learning purposes, it's helpful to keep indexes around to inspect in the Pinecone console


## Future Enhancements

- Add real embedding models (OpenAI, SentenceTransformers)
- Add real-world dataset examples
- Add performance benchmarking tools
- Add visualization tools for vector spaces
