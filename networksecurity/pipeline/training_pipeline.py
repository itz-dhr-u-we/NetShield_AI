import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

from networksecurity.entity.config_entity import (
    DataIngestConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from networksecurity.entity.artifact_entity import (
    DataIngestArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self):
        try:
            config = DataIngestConfig(self.training_pipeline_config)
            logging.info("Starting Data Ingestion")

            ingestion = DataIngestion(data_ingestion_config=config)
            artifact = ingestion.initiate_data_ingestion()

            logging.info(f"Data Ingestion completed: {artifact}")
            return artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestArtifact):
        try:
            config = DataValidationConfig(self.training_pipeline_config)

            validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=config,
            )

            logging.info("Starting Data Validation")
            return validation.initiate_data_validation()

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact):
        try:
            config = DataTransformationConfig(self.training_pipeline_config)

            transformation = DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=config,
            )

            return transformation.initiate_data_transformation()

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact):
        try:
            config = ModelTrainerConfig(self.training_pipeline_config)

            trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=config,
            )

            return trainer.initiate_model_trainer()

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def run_pipeline(self):
        try:
            ingestion_artifact = self.start_data_ingestion()

            validation_artifact = self.start_data_validation(
                ingestion_artifact
            )

            transformation_artifact = self.start_data_transformation(
                validation_artifact
            )

            model_trainer_artifact = self.start_model_trainer(
                transformation_artifact
            )

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)