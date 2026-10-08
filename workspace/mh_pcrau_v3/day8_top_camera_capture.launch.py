"""Day-8 wrapper: allow the lazy headless top camera renderer to warm up."""
import importlib.util
from pathlib import Path

SOURCE = Path(__file__).resolve().parent / "g1/gazebo_pilot_512_view.launch.py"

def generate_launch_description():
    spec = importlib.util.spec_from_file_location("day8_base_capture", SOURCE)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    description = module.generate_launch_description()
    handler = description.entities[17].event_handler
    capture = vars(handler)["_OnActionEventBase__actions_on_event"][0]
    parameters = vars(capture)["_Node__parameters"][0]
    changed = False
    for key in list(parameters):
        if len(key) == 1 and getattr(key[0], "text", None) == "capture_timeout_sec":
            parameters[key] = 150.0; changed = True
    if not changed: raise RuntimeError("capture_timeout_sec parameter not found")
    return description
