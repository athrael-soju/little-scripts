from app import app
import json

with open("openapi.json", "w") as f:
    json.dump(app.openapi(), f, indent=2)
