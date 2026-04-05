import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openenv.core.env_server import create_fastapi_app
from server.environment import CropDoctorEnvironment
from models import CropAction, CropObservation

app = create_fastapi_app(CropDoctorEnvironment, CropAction, CropObservation)

@app.get("/")
def root():
    return {"status": "ok", "env": "crop_doctor_env", "version": "1.0.0"}

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
