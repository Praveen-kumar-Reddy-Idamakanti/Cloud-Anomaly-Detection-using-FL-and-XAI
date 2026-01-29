"""
Option to disable early stopping for full 50 epochs training
"""

# To run full 50 epochs, modify this in training_pipeline_fixed.py:

# Option 1: Increase patience to 50 (effectively disable)
self.early_stopping = EarlyStopping(
    patience=50,  # Changed from 10 to 50
    min_delta=self.config.min_delta
)

# Option 2: Comment out early stopping check
# if self.early_stopping(val_loss, self.model):
#     logger.info(f"Early stopping triggered at epoch {epoch + 1}")
#     break

# Expected results with full 50 epochs:
# - Training time: ~25-30 minutes
# - Similar performance (early stopping already found optimal point)
# - More training curves for visualization
# - Potential slight overfitting
