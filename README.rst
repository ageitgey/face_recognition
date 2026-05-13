# Face Recognition FastAPI BY Jackson Kim

A modern real-time face recognition system built with:

- Python
- FastAPI
- OpenCV
- face_recognition

Supports:

- Webcam recognition
- Face labeling
- REST API
- Docker deployment

Deployment Options
Local
uvicorn app.main:app --reload
Docker
docker build -t face-api .
docker run -p 8000:8000 face-api
Cloud

Recommended:

Railway
Render
AWS EC2
DigitalOcean
Azure
Performance Tips
Resize frames before recognition
Use GPU acceleration
Cache embeddings
Use async processing
Reduce webcam resolution