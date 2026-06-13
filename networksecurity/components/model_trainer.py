import os
import sys

from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig



from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object,load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from dotenv import load_dotenv
load_dotenv()
import mlflow
from urllib.parse import urlparse

def setup_mlflow():
    if os.getenv("MLFLOW_TRACKING_URI"):
        os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")

    if os.getenv("MLFLOW_TRACKING_USERNAME"):
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")

    if os.getenv("MLFLOW_TRACKING_PASSWORD"):
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

    import dagshub
    dagshub.init(
        repo_owner="itz-dhr-u-we",
        repo_name="networksecurity",
        mlflow=True,
    )


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            setup_mlflow()
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, best_model, train_metric, test_metric):
        mlflow.set_registry_uri(
            "https://dagshub.com/itz-dhr-u-we/networksecurity.mlflow"
        )
        mlflow.set_experiment("NetworkSecurity")

        with mlflow.start_run():
            mlflow.log_metric("train_f1", train_metric.f1_score)
            mlflow.log_metric("test_f1", test_metric.f1_score)
            mlflow.log_metric("train_precision", train_metric.precision_score)
            mlflow.log_metric("test_precision", test_metric.precision_score)

            mlflow.sklearn.log_model(best_model, "model")

    def train_model(self, X_train, y_train, X_test, y_test):
        models = {
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "AdaBoost": AdaBoostClassifier(),
        }

        params = {
            "Random Forest": {"n_estimators": [8, 16, 32, 64, 128]},
            "Decision Tree": {
                "criterion": ["gini", "entropy", "log_loss"]
            },
            "Gradient Boosting": {
                "learning_rate": [0.1, 0.01, 0.05],
                "n_estimators": [8, 16, 32, 64],
            },
            "Logistic Regression": {},
            "AdaBoost": {
                "learning_rate": [0.1, 0.01],
                "n_estimators": [8, 16, 32],
            },
        }

        model_report = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            models=models,
            param=params,
        )

        best_model_name = max(model_report, key=model_report.get)
        best_model = models[best_model_name]

        # predictions
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        train_metric = get_classification_score(y_train, y_train_pred)
        test_metric = get_classification_score(y_test, y_test_pred)

        self.track_mlflow(best_model, train_metric, test_metric)

        preprocessor = load_object(
            file_path=self.data_transformation_artifact.transformed_object_file_path
        )

        # IMPORTANT: save final artifacts for deployment
        os.makedirs("final_model", exist_ok=True)

        save_object("final_model/model.pkl", best_model)
        save_object("final_model/preprocessor.pkl", preprocessor)

        # artifact object
        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path="final_model/model.pkl",
            train_metric_artifact=train_metric,
            test_metric_artifact=test_metric,
        )

        logging.info(f"Model trainer artifact: {model_trainer_artifact}")

        # save artifact for dashboard/metrics
        os.makedirs("artifact", exist_ok=True)
        save_object("artifact/model_trainer_artifact.pkl", model_trainer_artifact)

        return model_trainer_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            return self.train_model(X_train, y_train, X_test, y_test)

        except Exception as e:
            raise NetworkSecurityException(e, sys)