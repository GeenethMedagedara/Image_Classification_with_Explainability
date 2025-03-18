"""
Handles evaluation strategies for image classification models.
"""

import numpy as np
import matplotlib.pyplot as plt

class EvaluationStrategy:
    """Abstract class defining the evaluation strategy."""
    def evaluate(self, history):
        raise NotImplementedError("Subclasses must implement 'evaluate' method.")

class StandardEvaluation(EvaluationStrategy):
    """Concrete strategy for standard evaluation."""
    def evaluate(self, history):
        plt.figure(figsize=(12, 6))

        # Accuracy Plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss Plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

        # Print Best Validation Accuracy
        best_accuracy = np.max(history.history['val_accuracy'])
        print(f'Best Validation Accuracy = {best_accuracy:.4f}')
        return best_accuracy

class EvaluationContext:
    """Context class to execute different evaluation strategies."""
    def __init__(self, strategy: EvaluationStrategy):
        self.strategy = strategy

    def execute_evaluation(self, history):
        return self.strategy.evaluate(history)
