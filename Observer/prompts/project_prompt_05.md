# Step 1 - Adding a Deep Learning Anomaly Detector (PyTorch)

## Objective
* Define a new implementation of **IAnomalyDetector** named **PyTorchAutoencoderDetector**. 
* This detector must use a PyTorch neural network named **AutoencoderNet** to find anomalies based on kinematic state vectors.

## Training & Logging Requirements
* It must log its internal steps so the user/system can monitor progress.
* Provide an inline console progress bar during the training loop using `#` characters and a percentage from 0 to 100.
* **Callback Support:** The `fit()` method should accept an optional `progress_callback(percent: int)` so the Supervisor can report real-time training progress to the web UI (make it optional!)
  * Training should track two key parameters:
    * `error_function` (e.g., Mean Squared Error - MSE).
    * `maximum_normal_error` - this need to be calculated after training: 
  
  ```python  
  maximum_normal_error = float(
  np.percentile(error_per_sample, percentile_threshold)
              )
    
  
 * where `error_per_sample` error value calculated during training, `percentile_threshold` value in percent to steer percentil of error should be taken (usually 0.95 to 0.99) so the biggest errors during training will not be taken and this value is given as parameter. 

## Anomaly Detection Logic
* Anomaly detection in the `predict()` method must be evaluated by comparing the current frame's error (`output_error`) against the baseline maximum `maximum_normal_error`.
* Implement the following logic:
  ```python
  is_anomaly = output_error > maximum_normal_error
  return is_anomaly, output_error

## Testing
* Provide simple test of PyTorchAutoencoderDetector, so we can be sure that flow works correctly.

# Step 2 - Connect in UI
## Objective
* Make possible in UI (web page) to decide which implementation to use (isolation forest, or Autoencoder).
* When detection is started (SupervisorState.TRAINING) then selected implementation should be used.
* Training progress is shown as bar.
