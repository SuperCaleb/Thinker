# Thinker

Thinker is an advanced AI system designed to predict vehicle collisions from dashcam videos. Thinker leverages state-of-the-art techniques including feature extraction, Transformer-based sequence modeling, and robust data augmentation to handle real-world challenges effectively.

## How Thinker Works

Thinker operates by processing dashcam video frames to predict potential collisions. It uses a two-stream model architecture combining spatial and temporal information, extracted via EfficientNetB3 networks. The spatial stream analyzes individual frames, while the temporal stream processes optical flow between consecutive frames to capture motion information. These features are then fed into a Transformer encoder for sequence modeling.

### Key Components

1. **Spatial Stream**: Processes individual frames using EfficientNetB3 and extracts spatial features.
2. **Temporal Stream**: Computes optical flow between consecutive frames and extracts temporal features using another EfficientNetB3.
3. **Transformer Encoder**: Models temporal dependencies in the extracted features to predict collision probabilities.
4. **Prometheus Monitoring**: Provides real-time metrics on the number of videos processed, average collision probability, and alert levels.

### Workflow

1. **Frame Extraction**: Video frames are extracted and preprocessed.
2. **Feature Extraction**: Spatial and temporal features are extracted from the frames.
3. **Sequence Creation**: Sequences of features are created for prediction.
4. **Collision Prediction**: The model predicts collision probabilities and classifies the event type (normal, near-collision, collision).

## Example Outputs

When Thinker processes a video, it generates predictions for each frame indicating the likelihood of a collision. An example output might look like this:

```json
{
  "predictions": [0.03, 0.05, 0.07, ..., 0.95, 0.97, 0.99],
  "event_type": "collision",
  "event_time": 12.3,
  "alert_time": 10.5
}
```

- `predictions`: A list of collision probabilities for each frame.
- `event_type`: The type of event detected (`normal`, `near-collision`, `collision`).
- `event_time`: The time (in seconds) when the collision is detected.
- `alert_time`: The time (in seconds) when an alert is first raised.

## How to Use Thinker

### Prerequisites

- Python 3.7 or higher
- TensorFlow 2.x
- OpenCV
- Prometheus Client
- Email and Slack configurations (optional for notifications)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/SuperCaleb/Thinker.git
cd Thinker
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Configuration

Edit the `ThinkerConfig` class in `Thinker.py` to customize Thinker's behavior, such as logging, model weights directory, alert thresholds, and notification settings.

### Running Thinker

1. Start the Prometheus server (optional):

```bash
python -m prometheus_client
```

2. Run Thinker:

```bash
python Thinker.py
```

### Training the Model

Thinker can be trained using a dataset of labeled videos. Update the `video_paths` and `labels` variables with the paths to your training videos and their corresponding labels. Then run the training function:

```python
thinker = Thinker()
thinker.train(video_paths, labels, epochs=10)
```

### Health Check

Thinker performs periodic health checks to ensure the model is functioning correctly:

```python
async def health_check(thinker):
    # Health check logic
```

### Notifications

Thinker can send email and Slack notifications for detected collisions. Ensure the email and Slack configurations are set in the `ThinkerConfig` class.

## Conclusion

Thinker is a powerful AI system designed to assist in predicting vehicle collisions from dashcam videos. With its advanced architecture and robust feature extraction capabilities, Thinker can effectively handle real-world scenarios and provide timely alerts to prevent accidents.
