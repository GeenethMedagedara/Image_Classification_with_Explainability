from src.models.explainability import ExplainabilityStrategy  # Import the base class

class ExplainabilityContext:
    """Context class to execute different explainability strategies."""
    def __init__(self, strategy: ExplainabilityStrategy):
        self.strategy = strategy

    def execute_explanation(self, model, image):
        return self.strategy.explain(model, image)
