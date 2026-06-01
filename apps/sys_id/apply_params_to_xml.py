#!/usr/bin/env python3
"""
SpiRob – Übernimmt identifizierte Parameter in eine neue XML-Datei
"""

import json
from pathlib import Path
import mujoco as mj
import numpy as np

def set_params(model: mj.MjModel,
               stiffness: np.ndarray,
               damping: np.ndarray,
               tendon_stiffness: np.ndarray) -> None:
    """Sets individual physical parameters (arrays) for all joints and tendons."""
    for i in range(model.njnt):
        model.jnt_stiffness[i] = stiffness[i]
        dof = model.jnt_dofadr[i]
        model.dof_damping[dof] = damping[i]
    for i in range(model.ntendon):
        model.tendon_stiffness[i] = tendon_stiffness[i]

def main():
    base_dir = Path(__file__).resolve().parent
    xml_path = base_dir / "spiral_chain_wo_cylinder.xml"
    json_path = base_dir / "build" / "sysid_real_linear_profile_params.json"
    out_xml_path = base_dir / "spiral_chain_identified_linear.xml"

    if not xml_path.exists():
        print(f"Fehler: Basis-XML nicht gefunden unter {xml_path}")
        return
    if not json_path.exists():
        print(f"Fehler: JSON-Parameter-Datei nicht gefunden unter {json_path}")
        return

    print(f"Lade Basis-ML: {xml_path}")
    print(f"Lade Parameter: {json_path}")

    # Lade Parameter aus JSON
    with open(json_path, "r") as f:
        params_dict = json.load(f)
        
    stiffness = np.array(params_dict["stiffness"])
    damping = np.array(params_dict["damping"])
    tendon_stiffness = np.array(params_dict["tendon_stiffness"])

    # Lade MuJoCo Modell
    model = mj.MjModel.from_xml_path(str(xml_path))

    # Setze neue Parameter ins Modell ein
    set_params(model, stiffness, damping, tendon_stiffness)

    # Speichere als neue XML-Datei
    mj.mj_saveLastXML(str(out_xml_path), model)
    print("=" * 55)
    print(f"Erfolg! Modifiziertes Modell gespeichert unter:\n{out_xml_path}")

if __name__ == "__main__":
    main()
