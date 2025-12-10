import os
from pathlib import Path
import logging
from setuptools import find_packages, setup

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    f'src/__int__.py',
    f'src/components/__init__.py',
    f'src/components/data_ingestion.py',
    f'src/components/model_trainer.py',
    f'src/components/data_transformation.py',
    F'src/components/model_evaluation.py',
    f'src/pipeline/__init__.py',
    f'src/pipeline/train_pipeline.py',
    f'src/pipeline/predict_pipeline.py',
    f'src/utils/__init__.py',
    f'src/utils/utility.py',
    f'src/utils/logger.py',
    f'src/utils/exception.py',
    f'notebooks/experiment.ipynb',
    f'data/data.txt',
    f'artifacts/model.txt',
    f'templates/home.html',
    f'templates/index.html',
    f'main.py',
    f'template.py',
    f'README.md'
]

for filepath in list_of_files:
    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir != '':
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath, 'w') as file:
            pass
        logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} is already exists.")



setup(
    name = "Hand Written Digits Classification",
    version='0.0.1',
    author= 'Roben',
    author_email= 'creatoroben@gmail.com',
    packages=find_packages()
)