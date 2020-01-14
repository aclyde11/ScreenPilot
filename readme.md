##


1. Generate ligand 
| |
| |
| |
2. 2d sim to known kinase inhibitors (set used)
- crawl away from distribution

score = max(sim(this_ligand, i))

if score < delta:

3. ROCS to Kinase Inhibit
- generate conformers
- score  = max(sim(conf, j))
- j is in pdb structures 
- Which scores to use? TanimotoCombo

2. Check unique to and  