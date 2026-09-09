#!/usr/bin/env python
"""
Batch ESM3dG scorer for a library of PDB/CIF structures.

Reads a manifest (one structure path per line), scores each with the augmented
3-checkpoint ensemble, and writes a CSV. Supports optional single-point
mutational scanning and complex (binder+target) ΔG_AB ranking.

Sharding: pass --shard i/n so a SLURM array of n tasks each handles the files
where (line_index % n == i). Each shard writes its own CSV -> no write races.

Examples
--------
  # monomer ΔG for every apo structure, chain A
  python score_structures.py --manifest pyr1_apo.txt --outdir results --chain A

  # complex ΔG_AB ranking (HAB1 / binder designs; A=binder, B=target)
  python score_structures.py --manifest hab1_complexes.txt --outdir results \
         --mode complex --binder-chain A --target-chain B

  # single-point mutational scan; saves a .npy per structure
  python score_structures.py --manifest wt.txt --outdir scan --mode scan --chain A
"""
import argparse, csv, os, sys, time

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="text file, one PDB/CIF path per line")
    p.add_argument("--outdir", required=True)
    p.add_argument("--chain", default="A")
    p.add_argument("--mode", choices=["monomer", "complex", "scan"], default="monomer")
    p.add_argument("--binder-chain", default="A")
    p.add_argument("--target-chain", default="B")
    p.add_argument("--repo", default=os.environ.get("REPO", "/PATH/TO/REPO/")) #EDIT REPO PATH
    p.add_argument("--weights-dir", default=None,
                   help="dir with ESM3dG_weights_augmented_{1,2,3}_lora.ckpt (default: <repo>/esm3dg_weights)")
    p.add_argument("--shard", default="0/1", help="i/n : this task handles files where idx %% n == i")
    return p.parse_args()

def main():
    a = parse_args()
    sys.path.insert(0, a.repo)
    from ESM3dG import ESM3dG, ESM3dG_predict, ESM3dG_predict_complex
    import numpy as np

    wdir = a.weights_dir or os.path.join(a.repo, "esm3dg_weights")
    weights = [os.path.join(wdir, f"ESM3dG_weights_augmented_{i}_lora.ckpt") for i in (1, 2, 3)]
    for w in weights:
        if not os.path.exists(w):
            sys.exit(f"Missing checkpoint: {w}  (run download_weights.sh first)")

    i, n = (int(x) for x in a.shard.split("/"))
    with open(a.manifest) as f:
        files = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    mine = [f for k, f in enumerate(files) if k % n == i]
    os.makedirs(a.outdir, exist_ok=True)
    print(f"[shard {i}/{n}] {len(mine)}/{len(files)} structures | mode={a.mode}", flush=True)

    print("Loading 3-checkpoint augmented ensemble ...", flush=True)
    models = [ESM3dG(w) for w in weights]

    out_csv = os.path.join(a.outdir, f"monomer_scores_shard{i}_of{n}.csv")
    rows = []
    for k, path in enumerate(mine):
        t0 = time.time()
        name = os.path.basename(path)
        try:
            if a.mode == "monomer":
                vals = [ESM3dG_predict(m, path, a.chain)[1][0] for m in models]
                rows.append({"structure": name, "path": path,
                             "dG_ensemble": sum(vals) / len(vals),
                             "dG_ckpt1": vals[0], "dG_ckpt2": vals[1], "dG_ckpt3": vals[2]})
            elif a.mode == "complex":
                # Each model returns (dg_binder, dg_target, dg_complex). Keep all three:
                # dg_binder = monomer stability of chain A (binder) on its own,
                # dg_target = monomer stability of chain B, dg_AB = complex.
                trip = [ESM3dG_predict_complex(m, path, a.binder_chain, a.target_chain) for m in models]
                dgA  = [t[0] for t in trip]
                dgB  = [t[1] for t in trip]
                ab   = [t[2] for t in trip]
                mA, mB, mAB = (sum(x) / len(x) for x in (dgA, dgB, ab))
                rows.append({"structure": name, "path": path,
                             "dG_binder_ensemble": mA, "dG_target_ensemble": mB,
                             "dG_AB_ensemble": mAB,
                             # complexation proxy: change in mean per-residue stability
                             # (NOT a rigorous binding ΔΔG — see README use-case notes)
                             "dG_bind_delta": mAB - 0.5 * (mA + mB),
                             "dG_binder_ckpt1": dgA[0], "dG_binder_ckpt2": dgA[1], "dG_binder_ckpt3": dgA[2],
                             "dG_AB_ckpt1": ab[0], "dG_AB_ckpt2": ab[1], "dG_AB_ckpt3": ab[2]})
            elif a.mode == "scan":
                # ensemble-averaged single-point ΔΔG scan, shape (L,20,1,L) -> save .npy
                scans = [ESM3dG_predict(m, path, a.chain, ddg_scanning=True)[0].numpy() for m in models]
                seq = ESM3dG_predict(models[0], path, a.chain, ddg_scanning=True)[2]
                mean_scan = np.mean(scans, axis=0)
                np.save(os.path.join(a.outdir, f"{name}.ddgscan.npy"), mean_scan)
                rows.append({"structure": name, "path": path,
                             "seq_len": len(seq), "scan_npy": f"{name}.ddgscan.npy"})
            print(f"  [{k+1}/{len(mine)}] {name}  ({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"  [FAIL] {name}: {e}", flush=True)
            rows.append({"structure": name, "path": path, "error": str(e)})

    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
    print(f"Wrote {out_csv}", flush=True)

if __name__ == "__main__":
    main()
