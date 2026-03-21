import numpy as np, json
from pyscf import gto, scf, ao2mo
from openfermion import (jordan_wigner, get_fermion_operator, InteractionOperator,
                         count_qubits, get_sparse_operator)

def get_edges_from_mol(mol_obj, n_active, n_frozen=0, use_rohf=False):
    mf = (scf.ROHF(mol_obj) if use_rohf else scf.RHF(mol_obj)).run(verbose=0)
    n_orb = mol_obj.nao
    mo = mf.mo_coeff
    h1 = mo.T @ mf.get_hcore() @ mo
    eri = ao2mo.kernel(mol_obj, mo, compact=False).reshape(n_orb, n_orb, n_orb, n_orb)
    s = slice(n_frozen, n_frozen + n_active)
    h1a, h2a, nuc = h1[s, s], eri[s, s, s, s], mol_obj.energy_nuc()
    h2_of = np.zeros_like(h2a)
    for p in range(n_active):
        for q in range(n_active):
            for r in range(n_active):
                for t in range(n_active):
                    h2_of[p, q, r, t] = h2a[p, t, r, q]
    fop = get_fermion_operator(InteractionOperator(nuc, h1a, 0.5 * h2_of))
    qh = jordan_wigner(fop)
    n_q = count_qubits(qh)
    matrix = get_sparse_operator(qh, n_qubits=n_q).toarray()
    edges = []
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[0]):
            if abs(matrix[i, j]) > 1e-12:
                edges.append([format(i, f"0{n_q}b"), format(j, f"0{n_q}b")])
    return n_q, edges

results = {}

# Li3
mol = gto.M(atom="Li 0 0 0; Li 2.7 0 0; Li 1.35 2.34 0", basis="sto-3g", spin=1, verbose=0)
nq, e = get_edges_from_mol(mol, 3, 3, use_rohf=True)
results["Li3_3orb"] = {"title": "Li3 Triangle (3 active, STO-3G)", "nq": nq, "edges": e}

# H2O
mol = gto.M(atom="O 0 0 0; H 0.96 0 0; H -0.24 0.93 0", basis="sto-3g", verbose=0)
nq, e = get_edges_from_mol(mol, 3, 1)
results["H2O_3orb"] = {"title": "H2O (3 active, STO-3G)", "nq": nq, "edges": e}
nq, e = get_edges_from_mol(mol, 4, 1)
results["H2O_4orb"] = {"title": "H2O (4 active, STO-3G)", "nq": nq, "edges": e}

# BH3
mol = gto.M(atom="B 0 0 0; H 1.19 0 0; H -0.595 1.03 0; H -0.595 -1.03 0", basis="sto-3g", verbose=0)
nq, e = get_edges_from_mol(mol, 3, 1)
results["BH3_3orb"] = {"title": "BH3 (3 active, STO-3G)", "nq": nq, "edges": e}
nq, e = get_edges_from_mol(mol, 4, 1)
results["BH3_4orb"] = {"title": "BH3 (4 active, STO-3G)", "nq": nq, "edges": e}

# NH3
mol = gto.M(atom="N 0 0 0; H 0.94 0 0; H -0.47 0.81 0; H -0.47 -0.81 0", basis="sto-3g", verbose=0)
nq, e = get_edges_from_mol(mol, 3, 1)
results["NH3_3orb"] = {"title": "NH3 (3 active, STO-3G)", "nq": nq, "edges": e}
nq, e = get_edges_from_mol(mol, 4, 1)
results["NH3_4orb"] = {"title": "NH3 (4 active, STO-3G)", "nq": nq, "edges": e}

# H3+
mol = gto.M(atom="H 0 0 0; H 0.87 0 0; H 0.435 0.753 0", basis="sto-3g", charge=1, verbose=0)
nq, e = get_edges_from_mol(mol, 3, 0)
results["H3p"] = {"title": "H3+ Triangle (3 orb, STO-3G)", "nq": nq, "edges": e}

# H4 square
mol = gto.M(atom="H 0 0 0; H 1.0 0 0; H 1.0 1.0 0; H 0 1.0 0", basis="sto-3g", verbose=0)
nq, e = get_edges_from_mol(mol, 4, 0)
results["H4_square"] = {"title": "H4 Square (4 orb, STO-3G)", "nq": nq, "edges": e}

# LiH
mol = gto.M(atom="Li 0 0 0; H 1.6 0 0", basis="sto-3g", verbose=0)
nq, e = get_edges_from_mol(mol, 3, 1)
results["LiH_3orb"] = {"title": "LiH (3 active, STO-3G)", "nq": nq, "edges": e}
nq, e = get_edges_from_mol(mol, 4, 1)
results["LiH_4orb"] = {"title": "LiH (4 active, STO-3G)", "nq": nq, "edges": e}

# CH4 5 active
mol = gto.M(atom="C 0 0 0; H 0.63 0.63 0.63; H 0.63 -0.63 -0.63; H -0.63 0.63 -0.63; H -0.63 -0.63 0.63",
            basis="sto-3g", verbose=0)
nq, e = get_edges_from_mol(mol, 5, 1)
results["CH4_5orb"] = {"title": "CH4 (5 active, STO-3G)", "nq": nq, "edges": e}

for k, v in results.items():
    print(f"{v['title']:45s} {v['nq']}q {len(v['edges'])}E")

out_path = "examples/molecular_edges.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {out_path}")
