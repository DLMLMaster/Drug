# Drug

## Table of Contents
  * [Installation](#installation)
  * [Quick start](#quick-start)
  * [Features](#features)
  * [License](#license)
  * [Acknowledgements](#acknowledgements)
  * [Disclaimer](#disclaimer)
  * [Attribution](#attribution)

## Installation

Download using pip via pypi.

```bash
$ pip install Drug --upgrade
  or
$ pip install git+https://github.com/DLMLMaster/Drug.git
```
(Mac/homebrew users may need to use ``pip3``)

## Quick start
```python
 >>> from Drug.Embedding import DrugAPI
 >>> api = DrugAPI()
 >>> api.drugList(file_path(str), col_drug_name(str))
 >>> api.fetch(property(str))
 >>> api.saveCsv(file_path(str))
 
 >>> from Drug.Embedding import DrugProcess
 >>> drug = DrugProcess()
 >>> drug.load(file_path(str))
 >>> drug.preprocess(strategy(str))
 >>> drug.embedding(col_vector(str), col_smiles(str))
 >>> drug.matrix_scaler(col_vector(str))
 >>> drug.feature_scaler(col_vector(str))
 >>> drug.combined_matrix()
 >>> drug.saveNp(file_path_combine(str or None), file_path_matrix(str or None))
```

## Features
  * The first class leverages the PubChem API to read a list of drug names from a CSV file and collect structural information for each drug. 
    This information is obtained through API requests, stored in a data frame, and finally output as a CSV file.
  * The second class preprocesses the collected drug data into a form suitable for model training. 
    This process involves filling the data frame with missing values ​​and using RDKit to convert each drug's SMILES string into a vector representation of its molecular structure. Additionally, optional numerical data processing methods are provided that can be applied if required.

## License
  * This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
  * This project uses data from PubChem. PubChem is a registered trademark of the National Library of Medicine, National Center for Biotechnology Information.

## Disclaimer
  * This product uses the PubChem API but is not endorsed or certified by PubChem.

## Attribution
  * This project includes code generated by ChatGPT, a language model developed by OpenAI. 
  * The generated code has been used and modified in accordance with OpenAI's terms of service.