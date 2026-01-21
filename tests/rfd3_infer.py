import warnings
warnings.filterwarnings('ignore', module='atomworks')

# Shared utilities for visualization (from AtomWorks)
from atomworks.io.utils.visualize import view

from lightning.fabric import seed_everything
from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine

# Set seed for reproducibility
seed_everything(0)

# Configure RFD3 inference
config = RFD3InferenceConfig(
    specification={
        'length': 80,  # Generate 80-residue proteins
    },
    diffusion_batch_size=2,  # Generate 2 structures per batch
)

# Initialize engine and run generation
model = RFD3InferenceEngine(**config)
outputs = model.run(
    inputs=None,      # None for unconditional generation
    out_dir=None,     # None to return in memory (no file output)
    n_batches=1,      # Generate 1 batch
)

outputs.keys()

# Inspect RFD3 outputs and extract the generated structures
for idx, data in outputs.items():
    print(f"Batch {idx}: {len(data)} structure(s)")
    print(f"  Output type: {type(data[0]).__name__}")
    print(f"  AtomArray: {data[0].atom_array}")

# Extract the first generated structure for downstream use
first_key = next(iter(outputs.keys()))
atom_array = outputs[first_key][0].atom_array

# Visualize the generated structure
# view(atom_array)

from mpnn.inference_engines.mpnn import MPNNInferenceEngine

# Configure MPNN inference engine
# See mpnn.utils.inference.MPNN_GLOBAL_INFERENCE_DEFAULTS for all options
engine_config = {
    "model_type": "ligand_mpnn",  # or "protein_mpnn" for vanilla ProteinMPNN
    "is_legacy_weights": True,    # Required for now for ligand_mpnn and protein_mpnn
    "out_directory": None,        # Return results in memory
    "write_structures": False,
    "write_fasta": False,
}

# Configure per-input inference options
# See mpnn.utils.inference.MPNN_PER_INPUT_INFERENCE_DEFAULTS for all options
input_configs = [
    {
        "batch_size": 10,         # Generate 10 sequences per structure
        "remove_waters": True,
    }
]

# Run sequence design on the RFD3-generated backbone
model = MPNNInferenceEngine(**engine_config)
mpnn_outputs = model.run(input_dicts=input_configs, atom_arrays=[atom_array])

from biotite.structure import get_residue_starts
from biotite.sequence import ProteinSequence

# Extract and display the designed sequences
print(f"Generated {len(mpnn_outputs)} designed sequences:\n")

for i, item in enumerate(mpnn_outputs):
    res_starts = get_residue_starts(item.atom_array)
    # Convert 3-letter codes to 1-letter using Biotite
    seq_1letter = ''.join(
        ProteinSequence.convert_letter_3to1(res_name)
        for res_name in item.atom_array.res_name[res_starts]
    )
    print(f"Sequence {i+1}: {seq_1letter}")

from rf3.inference_engines.rf3 import RF3InferenceEngine
from rf3.utils.inference import InferenceInput


# Initialize RF3 inference engine
inference_engine = RF3InferenceEngine(ckpt_path='rf3', verbose=False)

# Create input from the MPNN-designed structure (first design)
# This re-folds the sequence to validate it adopts the intended structure
input_structure = InferenceInput.from_atom_array(atom_array, example_id="example_protein")
rf3_outputs = inference_engine.run(inputs=input_structure)

# Outputs: dict mapping example_id -> list[RF3Output] (multiple models per input)
print(f"Output keys: {rf3_outputs.keys()}")
print(f"Number of models for 'example_protein': {len(rf3_outputs['example_protein'])}")

# Extract the top-ranked prediction
rf3_output = rf3_outputs["example_protein"][0]

# Inspect RF3Output structure
print(f"RF3Output contains:")
print(f"  - atom_array: {len(rf3_output.atom_array)} atoms")
print(f"  - summary_confidences: {list(rf3_output.summary_confidences.keys())}")
print(f"  - confidences: {list(rf3_output.confidences.keys()) if rf3_output.confidences else None}")

# Visualize the predicted structure
# view(rf3_output.atom_array)

# Summary confidences: overall model quality metrics
summary = rf3_output.summary_confidences

print("=== Summary Confidences ===")
print(f"  Overall pLDDT:    {summary['overall_plddt']:.3f}")
print(f"  Overall PAE:      {summary['overall_pae']:.2f} A")
print(f"  Overall PDE:      {summary['overall_pde']:.3f}")
print(f"  pTM:              {summary['ptm']:.3f}")
print(f"  ipTM:             {summary.get('iptm', 'N/A (single chain)')}")
print(f"  Ranking score:    {summary['ranking_score']:.3f}")
print(f"  Has clash:        {summary['has_clash']}")

# Detailed per-atom/residue confidences
conf = rf3_output.confidences

print("=== Per-Atom/Residue Confidences ===")
print(f"  atom_plddts:      {len(conf['atom_plddts'])} values (one per atom)")
print(f"  atom_chain_ids:   {len(conf['atom_chain_ids'])} values")
print(f"  token_chain_ids:  {len(conf['token_chain_ids'])} values (one per residue)")
print(f"  token_res_ids:    {len(conf['token_res_ids'])} values")
print(f"  PAE matrix:       {len(conf['pae'])}x{len(conf['pae'][0])}")

# Preview first 10 atom pLDDT scores
import numpy as np
print(f"\nFirst 10 atom pLDDTs: {np.round(conf['atom_plddts'][:10], 2).tolist()}")

from biotite.structure import rmsd, superimpose
from atomworks.constants import PROTEIN_BACKBONE_ATOM_NAMES
import numpy as np

# Get structures for comparison
aa_generated = atom_array              # Original RFD3 backbone (from Section 1)
aa_refolded = rf3_output.atom_array    # RF3-predicted structure

# Filter to backbone atoms (N, CA, C, O)
bb_generated = aa_generated[np.isin(aa_generated.atom_name, PROTEIN_BACKBONE_ATOM_NAMES)]
bb_refolded = aa_refolded[np.isin(aa_refolded.atom_name, PROTEIN_BACKBONE_ATOM_NAMES)]

# Superimpose structures and calculate RMSD
bb_refolded_fitted, _ = superimpose(bb_generated, bb_refolded)
rmsd_value = rmsd(bb_generated, bb_refolded_fitted)

print(f"Backbone RMSD: {rmsd_value:.2f} A")
print(f"\nInterpretation: {'Excellent' if rmsd_value < 1.0 else 'Good' if rmsd_value < 2.0 else 'Moderate'} designability")


from atomworks.io.utils.io_utils import to_cif_file

# Export structures to CIF format for visualization in PyMOL/ChimeraX
to_cif_file(aa_generated, "generated.cif")
to_cif_file(aa_refolded, "refolded.cif")

print("Exported structures:")
print("  - generated.cif: Original RFD3 backbone")
print("  - refolded.cif:  RF3-predicted structure")