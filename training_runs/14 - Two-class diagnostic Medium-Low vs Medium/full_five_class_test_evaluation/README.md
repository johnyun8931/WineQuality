# Full five-class test evaluation

The saved two-output diagnostic model was scored after training on all 750 original test rows, without refitting. It can predict only Medium-Low and Medium. All Low, Medium-High, and High examples are therefore misclassified. The two probability columns are the model's two-class probabilities, not a calibrated five-class distribution. Five-class cross entropy is not reported. The original two-class training/validation scores remain in the parent directory.
