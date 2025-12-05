# Moca4Cup

## Usage
- Complexity Identification
- Hierarchical Demonstration Retrieval + In-Context-Learning Generation
- Result Evaluation

### Complexity Identification
#### Feature Extraction
In this project, the specific operations for feature extraction are mainly located in the **features** folder.
Among them, ChangeType, DependencyDegree, and RefactoringOperation need to rely on the tools ChangeDistiller, Soot, and RefactoringMiner.

In the **info** folder, we have provided a dataset containing information on all the features used and the results of complexity classification.
#### Partioning
Ours/rankComplexData.py
#### Classifier
Ours/mlClassifiers.py

### Hierarchical Demonstration Retrieval + In-Context-Learning Generation
`python ours/Moca4Cup.py --model "gemini-2.5-flash-nothinking"`

When updating comments, please first check whether the tree_sitter tool has been downloaded and the corresponding model's api-key has been configured. When using a local model, please modify the client_method information in the main method of the code. The project has provided a trained lightgbm.pkl for predicting complexity, and you can also replace it as needed.

Coarse-grained retrieval uses the FAISS vector database. Before updating, the vector database needs to be built, which is done using the **build_dual_faiss_indices** function in the Moca4Cup file.
### Result Evaluation
`python ours/eval_Moca4Cup.py`

## Requirements
```
nlg-eval 2.3
nltk 3.9.1
python 3.9
soot 4.3.0
scipy
lightgbm
pandas
numpy
openai
faiss
tree_sitter 
```




