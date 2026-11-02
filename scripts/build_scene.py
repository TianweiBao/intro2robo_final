"""Regenerate models/scene.xml from pipette_sim.scene_builder.SceneConfig and check that it compiles."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco  # noqa: E402

from pipette_sim.scene_builder import SceneConfig, write_scene  # noqa: E402

if __name__ == "__main__":
    path = write_scene(SceneConfig())
    model = mujoco.MjModel.from_xml_path(str(path))
    print(f"wrote {path}  (nq={model.nq}, nv={model.nv}, nu={model.nu}, nbody={model.nbody}, "
          f"ngeom={model.ngeom}, neq={model.neq}, ncam={model.ncam})")
