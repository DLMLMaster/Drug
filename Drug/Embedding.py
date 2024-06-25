"""
The project mainly targets two functions: drug data collection and preprocessing.

The first feature is the DrugAPI class. 
This class leverages the PubChem API to read a list of drug names from a CSV file and collect structural information for each drug. 
This information is retrieved through an API request, stored in a data frame, and finally output as a CSV file.

The second function belongs to the DrugProcess class. 
This class preprocesses the collected drug data into a form suitable for model training. 
This process involves filling in missing values in the data frame, using RDKit to convert each drug's SMILES string into a vector representation of its molecular structure. 
In addition, optional numerical data processing functions are provided so that they can be applied as needed.

The main goal of this project is to effectively vectorize SMILES data and use it as input to models.
"""


import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from sklearn.preprocessing import StandardScaler
import requests
import io
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DrugAPI:
    def __init__(self):
        """
        Initializes DrugAPI object with necessary attributes.
        """
        self.base_url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name'
        self.drug_list = []
        self.df = pd.DataFrame()
        self.session = requests.Session()

    def drugList(self, file_path, col_drug_name):
        """
        Loads a list of drug names from a CSV file.

        Args:
        - file_path (str): File path of the CSV containing drug names.
        - col_drug_name (str): Column name containing drug names.

        Returns:
        - List[str]: List of drug names.
        """
        df = pd.read_csv(file_path)
        self.drug_list = df[col_drug_name].tolist()
        return self.drug_list

    def fetch_method(self, drug_name, property):
        """
        Fetches specific properties of a drug from an API and appends to a DataFrame.

        Args:
        - drug_name (str): Name of the drug.
        - property (str): Property to fetch (e.g., CanonicalSMILES,IsomericSMILES -> when entering arguments, do not use spaces!).
        """
        url = f'{self.base_url}/{drug_name}/property/{property}/CSV'
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.text
            add_df = pd.read_csv(io.StringIO(data))
            self.df = pd.concat([self.df, add_df], ignore_index=True)
        except requests.HTTPError as e:
            logging.error(f'HTTP Error for {drug_name}: {e}')
        except Exception as e:
            logging.error(f'Error for {drug_name}: {e}')

    def fetch(self, property):
        """
        Fetches specified properties for all drugs in the drug list.

        Args:
        - property (str): Properties to fetch for each drug (comma-separated -> when entering arguments, do not use spaces!).

        Returns:
        - pd.DataFrame: DataFrame containing fetched data.
        """
        for drug_name in self.drug_list:
            self.fetch_method(drug_name, property)
            time.sleep(0.5)
        
        return self.df

    def saveCsv(self, file_path):
        """
        Saves the collected drug data to a CSV file.

        Args:
        - file_path (str): File path to save the CSV.
        """
        self.df.to_csv(file_path, index=False)


class DrugProcess:
    def __init__(self):
        """
        Initializes DrugProcess object with necessary attributes.
        """
        self.df = pd.DataFrame()
        self.copy = pd.DataFrame()
        self.scaled_feature_matrix = None
        self.numeric_matrix = None
        self.new_combine_matrix = None

    def load(self, file_path):
        """
        Loads drug data from a CSV file into a DataFrame.

        Args:
        - file_path (str): File path of the CSV containing drug data.

        Returns:
        - pd.DataFrame: Loaded DataFrame.
        """
        self.df = pd.read_csv(file_path)
        return self.df

    def preprocess(self, strategy):
        """
        Preprocesses the drug data by handling missing values.

        Args:
        - strategy (str): Strategy to fill missing values ('mean', 'median', 'mode').

        Returns:
        - pd.DataFrame: Preprocessed DataFrame.
        """
        for col in self.df.columns:
            if self.df[col].dtype != 'object':
                if self.df[col].isnull().sum() > 0:
                    if strategy == 'mean':
                        self.df[col] = self.df[col].fillna(self.df[col].mean())
                    elif strategy == 'median':
                        self.df[col] = self.df[col].fillna(self.df[col].median())
                    elif strategy == 'mode':
                        self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            else:
                if self.df[col].isnull().sum() > 0:
                    self.df[col] = self.df[col].fillna('')
        return self.df

    def embedding(self, col_vector, col_smiles):
        """
        Embeds SMILES strings into vector representations using RDKit.

        Args:
        - col_vector (str): Name of the column to store vector embeddings.
        - col_smiles (str): Name of the column containing SMILES strings.

        Returns:
        - pd.DataFrame: DataFrame with added vector embedding column.
        """
        self.df[col_vector] = self.df[col_smiles].apply(self.embedding_method)
        return self.df

    @staticmethod
    def embedding_method(smiles):
        """
        Computes Morgan fingerprint vector for a given SMILES string.

        Args:
        - smiles (str): SMILES string of a molecule.

        Returns:
        - np.array: Morgan fingerprint vector (or zeros if SMILES is invalid).
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            return np.array(vect)
        else:
            return np.zeros(1024)

    def matrix_scaler(self, col_vector):
        """
        Scales the matrix of vector embeddings using StandardScaler.

        Args:
        - col_vector (str): Name of the column containing vector embeddings.

        Returns:
        - np.array: Scaled feature matrix.
        """
        feature_matrix = self.matrix_method(col_vector)
        scaler = StandardScaler()
        self.scaled_feature_matrix = scaler.fit_transform(feature_matrix)
        return self.scaled_feature_matrix

    def matrix_method(self, col_vector):
        """
        Retrieves the matrix representation from a DataFrame column.

        Args:
        - col_vector (str): Name of the column containing vector embeddings.

        Returns:
        - np.array: Matrix representation of vector embeddings.
        """
        return np.array(self.df[col_vector].tolist())

    def feature_scaler(self, col_vector):
        """
        Scales numeric features in the DataFrame using StandardScaler.

        Args:
        - col_vector (str): Name of the column containing vector embeddings.

        Returns:
        - pd.DataFrame: DataFrame with scaled numeric features.
        """
        self.copy = self.df.drop(col_vector, axis=1)
        numeric_cols = self.copy.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        self.copy[numeric_cols] = scaler.fit_transform(self.copy[numeric_cols])
        return self.copy

    def combined_matrix(self):
        """
        Combines scaled feature matrix with numeric feature matrix.

        Returns:
        - np.array: Combined matrix of scaled and numeric features.
        """
        numeric_cols = self.copy.select_dtypes(include=[np.number]).columns
        self.numeric_matrix = self.copy[numeric_cols].values
        self.new_combine_matrix = np.hstack((self.numeric_matrix, self.scaled_feature_matrix))
        return self.new_combine_matrix

    def saveNp(self, file_path_combine, file_path_matrix):
        """
        Saves matrices to numpy files.

        Args:
        - file_path_combine (str or None): File path to save combined matrix.
        - file_path_matrix (str or None): File path to save scaled feature matrix.
        """
        if self.new_combine_matrix is not None:
            np.save(file_path_combine, self.new_combine_matrix)
        else:
            np.save(file_path_matrix, self.scaled_feature_matrix)
