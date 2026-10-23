# Model Selection

## Primary metric
Validation average precision, selected before model comparison.

## Results
- Random Forest: 0.6700
- Logistic Regression: 0.6475
- Decision Tree: 0.6033
- Majority baseline: 0.2654

## Selected model
Random Forest was selected because it achieved the highest validation
average precision.

## Trade-offs
Logistic Regression achieved the highest accuracy (0.8041).
The Decision Tree achieved the highest recall (0.6016) and F1 score
(0.6156) at the default classification threshold of 0.5.

## Test set
The test set remains reserved for final evaluation.
