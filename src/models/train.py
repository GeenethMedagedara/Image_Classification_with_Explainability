"""
Handles training strategies for image classification models.
"""

import os
import mlflow
import mlflow.keras
import tensorflow as tf

mlflow.set_tracking_uri("http://localhost:5000")

class TrainingStrategy:
    def train(self, model, x_train, y_train, x_test, y_test):
        raise NotImplementedError("Subclasses must implement train method")

class StandardTraining(TrainingStrategy):
    def train(self, model, x_train, y_train, x_test, y_test):
        model_dir = "saved_model"
        logs_dir = "logs"
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        mlflow.set_experiment("Image Classification With Explainability")
        with mlflow.start_run():
            history = model.fit(
                x_train, y_train,
                validation_data=(x_test, y_test),
                batch_size=64,
                epochs=4,
                callbacks=[
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath=os.path.join(model_dir, "best_model.h5"),
                        save_best_only=True
                    ),
                    tf.keras.callbacks.TensorBoard(log_dir=logs_dir)
                ]
            )
            
            # Save model locally
            model.save(os.path.join(model_dir, "final_model.h5"))
            
            # Log parameters and metrics in MLflow
            mlflow.log_param("batch_size", 64)
            mlflow.log_param("epochs", 4)
            mlflow.log_metric("final_train_accuracy", history.history['accuracy'][-1])
            mlflow.log_metric("final_val_accuracy", history.history['val_accuracy'][-1])
            mlflow.keras.log_model(model, "model")
        
        return history

class TrainingContext:
    def __init__(self, strategy: TrainingStrategy):
        self.strategy = strategy
    
    def execute_training(self, model, x_train, y_train, x_test, y_test):
        return self.strategy.train(model, x_train, y_train, x_test, y_test)
