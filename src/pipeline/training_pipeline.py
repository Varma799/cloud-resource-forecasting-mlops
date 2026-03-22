from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation


def run_training_pipeline():
    ingestion = DataIngestion()
    validation = DataValidation()
    transformation = DataTransformation()
    trainer = ModelTrainer()
    evaluator = ModelEvaluation()

    print("Starting training pipeline...")

    ingested_path = ingestion.initiate_data_ingestion()
    print(f"Ingested data saved at: {ingested_path}")

    validation_status = validation.validate_data()
    print(f"Validation status: {validation_status}")

    transformed_path = transformation.initiate_data_transformation()
    print(f"Transformed data saved at: {transformed_path}")

    model_path = trainer.initiate_model_training()
    print(f"Best model saved at: {model_path}")

    evaluation_path = evaluator.initiate_model_evaluation()
    print(f"Evaluation report saved at: {evaluation_path}")

    print("Training pipeline completed successfully.")


if __name__ == "__main__":
    run_training_pipeline()