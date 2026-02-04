# Lab: Model Training

Train ML models with gradient descent and evaluate performance.

## Objectives

- Implement linear regression
- Train on synthetic datasets
- Calculate evaluation metrics

## Demo Code

See [`demos/course3/week3/model-training/`](https://github.com/noahgift/DB-mlops-genai/tree/main/demos/course3/week3/model-training)

## Lab Exercise

See [`labs/course3/week3/lab_3_4_automl.py`](https://github.com/noahgift/DB-mlops-genai/tree/main/labs/course3/week3)

## Key Implementation

```rust
impl LinearRegression {
    pub fn fit(&mut self, features: &[Vec<f64>], labels: &[f64]) {
        for _ in 0..self.n_iterations {
            let mut weight_gradients = vec![0.0; self.weights.len()];
            let mut bias_gradient = 0.0;

            for (x, &y) in features.iter().zip(labels.iter()) {
                let pred = self.predict_single(x);
                let error = pred - y;
                for (j, &xj) in x.iter().enumerate() {
                    weight_gradients[j] += error * xj;
                }
                bias_gradient += error;
            }

            // Update weights
            for (w, grad) in self.weights.iter_mut().zip(&weight_gradients) {
                *w -= self.learning_rate * grad / n_samples;
            }
            self.bias -= self.learning_rate * bias_gradient / n_samples;
        }
    }
}
```
