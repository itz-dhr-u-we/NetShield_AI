from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestConfig,DataValidationConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig
from networksecurity.components.data_validation import DataValidation


import sys


if __name__ == "__main__":
    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestConfig(trainingpipelineconfig)
        data_ingestion= DataIngestion(dataingestionconfig)
        
        logging.info("Initiate the data ingestion")
        dataingestionartifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data initiation completed.")
        print(dataingestionartifact)
        datavalidationconfig = DataValidationConfig(trainingpipelineconfig)
        data_validation = DataValidation(dataingestionartifact,datavalidationconfig)
        logging.info("Initiate the data validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data Validation completed.")
        print(data_validation_artifact)
    
    except Exception as e:
        raise NetworkSecurityException(e,sys)