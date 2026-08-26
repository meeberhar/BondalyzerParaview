import os
from paraview.simple import *

vtm_path = os.path.abspath("ethene_1d_zones.vtm")
reader = XMLMultiBlockDataReader(registrationName="ethene_1d", FileName=[vtm_path])
UpdatePipeline()

# Merge Blocks for Atoms (C) and Atoms (H)
extract_blocks = ExtractBlock(registrationName="Atoms_Block", Input=reader)
extract_blocks.Selectors = ['//*[@name="Atoms (C)"]', '//*[@name="Atoms (H)"]']
UpdatePipeline()

merged = MergeBlocks(registrationName="MergedAtoms", Input=extract_blocks)
UpdatePipeline()

# Apply ConvertIntoMolecule (ParaView's 'Convert Into Molecule' / 'Convert to Molecule' filter)
mol = ConvertIntoMolecule(registrationName="Molecule", Input=merged)
# In ParaView servermanager, the property name for input array selection is AtomicNumbers:
mol.AtomicNumbers = ['POINTS', 'atomic_number']
UpdatePipeline()

vtk_mol = servermanager.Fetch(mol)
print("VTK Molecule Number of Atoms:", vtk_mol.GetNumberOfAtoms() if vtk_mol else None)
print("VTK Molecule Number of Bonds:", vtk_mol.GetNumberOfBonds() if vtk_mol else None)

if vtk_mol:
    for i in range(vtk_mol.GetNumberOfAtoms()):
        atom = vtk_mol.GetAtom(i)
        print(f"  Atom {i}: Atomic Number = {atom.GetAtomicNumber()}, Position = {atom.GetPosition()}")

# Also compute bonds in ParaView
bonds = ComputeMoleculeBonds(registrationName="Bonds", Input=mol)
UpdatePipeline()
vtk_bonds = servermanager.Fetch(bonds)
print("\nAfter ComputeMoleculeBonds:")
print("Number of Atoms:", vtk_bonds.GetNumberOfAtoms())
print("Number of Bonds:", vtk_bonds.GetNumberOfBonds())
for b in range(vtk_bonds.GetNumberOfBonds()):
    bond = vtk_bonds.GetBond(b)
    print(f"  Bond {b}: Atom {bond.GetBeginAtomId()} (Z={vtk_bonds.GetAtom(bond.GetBeginAtomId()).GetAtomicNumber()}) <-> Atom {bond.GetEndAtomId()} (Z={vtk_bonds.GetAtom(bond.GetEndAtomId()).GetAtomicNumber()}), Order = {bond.GetOrder()}")
