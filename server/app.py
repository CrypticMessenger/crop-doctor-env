import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openenv.core.env_server import create_fastapi_app
from server.environment import CropDoctorEnvironment
from models import CropAction, CropObservation

app = create_fastapi_app(CropDoctorEnvironment, CropAction, CropObservation)
