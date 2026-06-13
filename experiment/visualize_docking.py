"""对接结果可视化 — py3Dmol 渲染蛋白-配体复合物.

用法:
    python visualize_docking.py experiment/runs_dhfr/seed_42/r01/eval.json

或 Jupyter:
    from visualize_docking import show_complex
    show_complex("runs_dhfr/seed_42/r01/eval.json", mol_index=0)
"""

import json, sys


def show_complex(eval_path, protein_pdb="experiment/1DLS_prepared.pdb", mol_index=0):
    """用 py3Dmol 在 Jupyter 中展示对接复合物."""
    import py3Dmol

    with open(eval_path) as f:
        mols = json.load(f)
    mol = mols[mol_index]
    pdbqt = mol.get("best_pose_pdbqt", "")
    smiles = mol.get("smiles", "?")
    score = mol.get("best_score", "?")

    # 蛋白
    with open(protein_pdb) as f:
        prot = f.read()

    view = py3Dmol.view(width=800, height=600)
    view.addModel(prot, "pdb")
    view.setStyle({"cartoon": {"color": "lightgray"}})
    # 结合位点高亮
    view.addStyle({"within": {"distance": 5, "sel": {"resn": ["UNL"]}}},
                  {"stick": {"colorscheme": "cyanCarbon"}})
    # 配体
    if pdbqt:
        view.addModel(pdbqt, "pdb")
        view.setStyle({"model": 1}, {"stick": {"colorscheme": "greenCarbon"}})
    view.zoomTo()
    return view


def render_png(eval_path, out_path, protein_pdb="experiment/1DLS_prepared.pdb", mol_index=0):
    """渲染对接复合物为 PNG 图片 (需要 py3Dmol + IPython)."""
    view = show_complex(eval_path, protein_pdb, mol_index)
    view.render(filename=out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python visualize_docking.py <eval.json> [mol_index]")
        sys.exit(1)
    path = sys.argv[1]
    idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    with open(path) as f:
        mols = json.load(f)
    mol = mols[idx]
    print(f"SMILES: {mol.get('smiles','')}")
    print(f"Score: {mol.get('best_score','?')} kcal/mol")
    print(f"Interactions: {mol.get('interactions','')}")
    print(f"PDBQT pose: {'saved' if mol.get('best_pose_pdbqt') else 'MISSING'}")
