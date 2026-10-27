# Training-subset memorization check

This run uses 64 original training examples, balanced 32 per class, and no dropout or L2. Validation-based early stopping is disabled. Training stops at 100% accuracy with cross entropy at most 0.01 for five consecutive epochs, or after 2,000 epochs. The original validation and test sets are not transformed or evaluated. Perfect training performance confirms the network can fit these training examples; it does not show that predictions generalize. The main experiment defaults are unchanged.
