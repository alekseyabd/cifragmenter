from __future__ import annotations
import logging
import signal
from pathlib import Path
import os
import uuid
import re
import pandas as pd
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import (
    SimplestChemenvStrategy,
)
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import (
    LocalGeometryFinder,
)
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import (
    LightStructureEnvironments,
)
from pymatgen.ext.matproj import MPRester
from pymatgen.core.structure import Structure, Molecule, Neighbor
from pymatgen.core.periodic_table import Element, ElementBase
from pymatgen.analysis.local_env import VoronoiNN, StructureGraph
from pymatgen.analysis.graphs import StructureGraph, MoleculeGraph
from pymatgen.analysis.local_env import CrystalNN, NearNeighbors, CovalentBondNN, BrunnerNN_real, BrunnerNN_reciprocal, BrunnerNN_relative, JmolNN
from pymatgen.analysis.chemenv.connectivity.connected_components import ConnectedComponent
from pymatgen.analysis.chemenv.connectivity.structure_connectivity import StructureConnectivity
from pymatgen.core.composition import Composition
from pymatgen.analysis.structure_analyzer import VoronoiAnalyzer
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.operations import SymmOp
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen.io.cif import CifParser
import openbabel
from pprint import pprint
from openbabel import pybel
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN

import sys
import numpy as np
import shutil
import math

from rdkit import Chem
from rdkit.Chem import AllChem, MolStandardize
from rdkit.Chem import Descriptors, Draw
from rdkit.Chem import rdDetermineBonds
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
from copy import deepcopy

import networkx as nx
from collections import Counter
ptable = Chem.GetPeriodicTable()

import zipfile
import tempfile
import json
from rich import print
from rich.table import Table
from rich.progress import track


PROJECT_ROOT = Path.cwd()

log = logging.getLogger("cifragmenter")
from .logging_conf import setup_logger
import traceback
logger = setup_logger('logfile', 'logfile.log')

def extract_elements(atom_list):
    elements = {re.match(r'[A-Za-z]+', atom).group() for atom in atom_list}
    return sorted(elements)

def get_species_element(site):
    if hasattr(site, 'species'):
        for el, occ in site.species.items():
            elem = el
            return elem
    else:
        elem = site.specie.symbol
        return elem

def get_voronoi_polyhedron_angles(structure, metal_index, cutoff=10.0):
    #elements=[Element('N'),Element('O'),Element('Cl'),Element('S'), Element('P'), Element('C'), Element('Se'), Element('F'), Element('Br'), Element('I'), Element('Si'), Element('B')]
    elements = ['N','Cl','S','P','C','Se','F','Br','I','Si','B']
    #vnn = VoronoiNN(cutoff=cutoff, compute_adj_neighbors=False, tol=0.3, targets=elements)
    vnn = VoronoiNN(cutoff=cutoff, compute_adj_neighbors=False)
    nn_info = vnn.get_nn_info(structure, metal_index)

    poly_info = vnn.get_voronoi_polyhedra(structure, metal_index)
    polyelements = [i['site'].label for i in poly_info.values()]
    total_volume = sum(info['volume'] for info in poly_info.values()) if poly_info else 0.0
    neighbor_angles = {}
    neighbor_distances = {}
    for neighbor in nn_info:
        site_index = neighbor['site_index']
        if site_index >= len(structure):
            continue
        elem = get_species_element(structure[site_index])
        if structure[site_index].label in polyelements and str(elem) in elements:
            if neighbor['poly_info']['face_dist']*2 < 3:
                angle = neighbor['poly_info']['solid_angle']
                neighbor_angles[site_index] = str((angle*100)/(4*3.14))+'_'+str(neighbor['poly_info']['face_dist']*2)
    return neighbor_angles, total_volume

def extract_submolecule(mol, indices):
    species = [mol[i].specie for i in indices]
    coords = [mol[i].coords for i in indices]
    site_properties = {}
    if hasattr(mol, 'site_properties'):
        for k, v in mol.site_properties.items():
            site_properties[k] = [v[i] for i in indices]
    return Molecule(species, coords, site_properties=site_properties)

def molgraph_to_rdkit(mol_graph, ligand):
    pmg_mol = mol_graph.molecule
    rd_mol = Chem.RWMol()
    for site in pmg_mol:
        rd_mol.AddAtom(Chem.Atom(site.specie.symbol))
    for i, j in mol_graph.graph.edges():
        rd_mol.AddBond(i, j, Chem.BondType.SINGLE)  
    rd_mol = rd_mol.GetMol()
    conf = Chem.Conformer(pmg_mol.num_sites)
    for i, site in enumerate(pmg_mol):
        conf.SetAtomPosition(i, [float(x) for x in site.coords])
    rd_mol.AddConformer(conf)
    
    if ligand==True:
        SA_str = []
        Dist_str = []
        Met_str = []
        Vpvd_str = []
        for i in range(pmg_mol.num_sites):
            atom = rd_mol.GetAtomWithIdx(i)
            metals = pmg_mol.site_properties.get("metals", [[]] * pmg_mol.num_sites)[i]
            sas = pmg_mol.site_properties.get("solid_angles", [[]] * pmg_mol.num_sites)[i]
            dists = pmg_mol.site_properties.get("distances", [[]] * pmg_mol.num_sites)[i]
            vols = pmg_mol.site_properties.get("poly_volumes", [[]] * pmg_mol.num_sites)[i]
            atom.SetProp("Metals", json.dumps(metals))
            atom.SetProp("SolidAngles", json.dumps(sas))
            atom.SetProp("Distances", json.dumps(dists))
            atom.SetProp("PolyVolumes", json.dumps(vols))
            SA_str.append(json.dumps(sas))
            Dist_str.append(json.dumps(dists))
            Met_str.append(json.dumps(metals))
            Vpvd_str.append(json.dumps(vols))
        rd_mol.SetProp("SA",str(SA_str))
        rd_mol.SetProp("Dist",str(Dist_str))
        rd_mol.SetProp("Metals",str(Met_str))
        rd_mol.SetProp("PolyVolumes",str(Vpvd_str))
        try:
            Chem.SanitizeMol(rd_mol)
            return rd_mol
        except:
            return None
    else:
        try:
            Chem.SanitizeMol(rd_mol)
            rw_mol = Chem.RWMol(rd_mol)
            for bond in rw_mol.GetBonds():
                if bond.GetBondType() == Chem.BondType.DATIVE:
                    bond.SetBondType(Chem.BondType.SINGLE)
            return rw_mol.GetMol()
        except:
            return None
        
def remove_duplicates_and_substructures(all_molecules, cif_name, db_code, chemical_name_systematic, property,first_name):
    try:
        rows = []
        for num, mol in enumerate(all_molecules):
            smiles = Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=True)
            name = f"db_code: {db_code} / cif: {cif_name} / num: {num} / name: {chemical_name_systematic} / smiles: {smiles} / property: {property}"
            mol.SetProp("_Name",name)
            mol.SetProp("cif_name",cif_name)
            mol.SetProp("compound_name",chemical_name_systematic)
            mol.SetProp("property", property)
            rows.append({
                "cif_name": first_name,
                "mol": mol,
                "smiles": smiles,
                "NumAt": mol.GetNumHeavyAtoms(),
                "db_code": db_code,
                "chemical_name_systematic": chemical_name_systematic,
                "property": property,
            })
        
        df = pd.DataFrame(rows)
        
        df = df.sort_values('NumAt', ascending=False, ignore_index=True)
        df = df.drop_duplicates(subset=['smiles','cif_name'], keep='first').reset_index(drop=True)
        mols = df["mol"].tolist()
        patterns = []

        for mol in mols:
            mol_no_h = Chem.RemoveHs(mol, sanitize=False)
            smarts = Chem.MolToSmarts(mol_no_h)
            patterns.append(Chem.MolFromSmarts(smarts))

        keep = [True] * len(mols)
        for i in range(1, len(mols)):
            patt = patterns[i]
            for large_mol in mols[:i]:
                if large_mol.HasSubstructMatch(patt):
                    keep[i] = False
                    break

        df = df.loc[keep].reset_index(drop=True)
        return(df)
    except:
        df = pd.DataFrame(columns=["cif_name", "mol", "smiles", "NumAt"])
        return(df)

def serach_name_ccdc(file_name,ccdc_chemical_name_systematic,db_code_pattern,property):
    lines = []
    with open(file_name, 'r', encoding='utf-8') as cif_file:
        for line in cif_file:
            lines.append(line)
    chemical_name_systematic=""
    db_code = ""
    property_arr = ""
    for n, i in enumerate(lines):
        if ccdc_chemical_name_systematic in i:
            if chemical_name_systematic != "":
                chemical_name_systematic = chemical_name_systematic+"//"
            if lines[n+1][0]!='_':
                z=2
                while z>1:
                    st=lines[n+z]
                    if ";" in st: z=0
                    else:
                        chemical_name_systematic=chemical_name_systematic+st.replace("\n", "").replace("  ", "")
                        z+=1
            else:
                chemical_name_systematic=chemical_name_systematic+i.split(str(i.split(' ')[0]))[1].replace("\n", "").replace("  ", "")
        if db_code_pattern in i:
            db_code = ''.join(i.strip().split()[1:])
        if property in i:
            property_arr=''.join(i.strip().split()[1:])
    return(chemical_name_systematic,db_code,property_arr)

def OpenBabelToMolGraph(file_name, xyz_lines,species,coords,supercell):
    try:
        xyz_text = "\n".join(xyz_lines)
        ob_mol = pybel.readstring("xyz", xyz_text)
        bonds = {}
        for bond in pybel.ob.OBMolBondIter(ob_mol.OBMol):
            i = bond.GetBeginAtomIdx() - 1
            j = bond.GetEndAtomIdx() - 1
            bonds[(i, j)] = None

        molecule = Molecule(species, coords)
        if hasattr(supercell, 'site_properties') and 'metals' in supercell.site_properties:
            molecule.add_site_property("metals", supercell.site_properties["metals"])
            molecule.add_site_property("solid_angles", supercell.site_properties["solid_angles"])
            molecule.add_site_property("distances", supercell.site_properties["distances"])
            molecule.add_site_property("poly_volumes", supercell.site_properties["poly_volumes"])
        mg = MoleculeGraph.from_edges(molecule=molecule, edges=bonds)
        G = nx.Graph(mg.graph)
        components = list(nx.connected_components(G))

        fragment_graphs = []
        for comp in components:
            nodes = sorted(comp)
            sub_G = nx.subgraph(G, nodes)
            sub_mol = extract_submolecule(molecule, nodes)
            global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(nodes)}
            local_edges = [(global_to_local[i], global_to_local[j]) for i, j in sub_G.edges()]
            edges = {}
            for i in local_edges:
                edges[i]= None
            frag_mg = MoleculeGraph.from_edges(
                molecule=sub_mol,
                edges=edges
            )
            fragment_graphs.append(frag_mg)
        if len(fragment_graphs)>1:
            return(fragment_graphs)
        else:
            fragment_graphs = []
            return(fragment_graphs)
    except Exception as e:
        logger.info(traceback.format_exc())
        if type(e).__name__ == "TimeoutException":
            fragment_graphs = "TimeoutException"
        else:
            fragment_graphs = []
        return(fragment_graphs)

def remove_low_occupancy_sites(structure, threshold=0.5):
    """
    Удаляет сайты с суммарным occupancy ниже порога
    """
    filtered_sites = []
    for site in structure.sites:
        occu = site.species.as_dict()[list(site.species.as_dict().keys())[0]]
        # Получаем суммарную занятость сайта
        total_occupancy = occu
        
        if total_occupancy >= threshold:
            filtered_sites.append(site)
    
    new_structure = Structure.from_sites(filtered_sites)
    return new_structure

def cif_to_mols(file_name, first_name, ccdc_chemical_name_systematic, db_code_pattern, min_occ,fragment_type,property):
    chemical_name_systematic,db_code,db_property = serach_name_ccdc(file_name,ccdc_chemical_name_systematic,db_code_pattern,property)
    
    try:
        CifPars = CifParser(file_name, occupancy_tolerance=6)
        struct = CifPars.get_structures()[0]
        supercell = struct.merge_sites(tol=0.1,mode="sum").make_supercell(2,2,2)

        if fragment_type == 'coord':
            for num,i in enumerate(struct):
                site = str(i)
                pattern = r"\] ([a-zA-Z]+)"
                match = re.search(pattern, site)
                if(Element(match.group(1)).is_metal):
                    struct[num].properties['metall'] = num
            SA=[0 for x in range(len(supercell))]
            Dist=[0 for x in range(len(supercell))]
            Met=[0 for x in range(len(supercell))]
            Vpvd = [0 for x in range(len(supercell))]
            donor_labels = {}
            
            for i, site in enumerate(supercell):
                elem = get_species_element(site)
                if(elem.is_metal):
                    metal_symbol = elem
                    neighbor_angles, poly_volume = get_voronoi_polyhedron_angles(supercell, i)
                    for nbr_idx, angle in neighbor_angles.items():
                        donor_atom = get_species_element(supercell[nbr_idx])
                        label = f"{metal_symbol}_{angle}_{poly_volume}"
                        if nbr_idx not in donor_labels:
                            donor_labels[nbr_idx] = []
                        donor_labels[nbr_idx].append(label)   
            for key,val in donor_labels.items():
                Vpvd[key]=float(val[0].split('_')[3])
                Dist[key]=float(val[0].split('_')[2])
                SA[key]=float(val[0].split('_')[1])
                Met[key]=val[0].split('_')[0]
            supercell.add_site_property("SA", SA)
            supercell.add_site_property("Dist", Dist)
            supercell.add_site_property("Vpvd", Vpvd)
            supercell.add_site_property("Met", Met)
            all_metals = [[] for _ in range(len(supercell))]
            all_solid_angles = [[] for _ in range(len(supercell))]
            all_distances = [[] for _ in range(len(supercell))]
            all_volumes = [[] for _ in range(len(supercell))]
            for donor_idx, labels in donor_labels.items():
                for label in labels:
                    metal_sym, sa_str, dist_str, vol_str = label.split('_')
                    all_metals[donor_idx].append(metal_sym)
                    all_solid_angles[donor_idx].append(float(sa_str))
                    all_distances[donor_idx].append(float(dist_str))
                    all_volumes[donor_idx].append(float(vol_str))
        
            metal_indices = [i for i, site in enumerate(supercell) if get_species_element(site).is_metal]
            nonmetal_structure = supercell.copy()
            nonmetal_structure.remove_sites(metal_indices)
            
            nonmetal_old_indices = [i for i in range(len(supercell)) if i not in metal_indices]            
            new_metals = [all_metals[i] for i in nonmetal_old_indices]
            new_solid_angles = [all_solid_angles[i] for i in nonmetal_old_indices]
            new_distances = [all_distances[i] for i in nonmetal_old_indices]
            new_volumes = [all_volumes[i] for i in nonmetal_old_indices]
            
            nonmetal_structure.add_site_property("metals", new_metals)
            nonmetal_structure.add_site_property("solid_angles", new_solid_angles)
            nonmetal_structure.add_site_property("distances", new_distances)
            nonmetal_structure.add_site_property("poly_volumes", new_volumes)
            supercell = nonmetal_structure
        supercell = remove_low_occupancy_sites(supercell)
        xyz_lines = [""]
        species = []
        coords = []
        for site in supercell:
            del_site = False
            if hasattr(site, 'species'):
                for el, occ in site.species.items():
                    elem = el
                    if occ < min_occ:
                        del_site = True
                if del_site == True:
                    continue
            else:
                elem = site.specie.symbol
            x, y, z = site.coords
            xyz_lines.append(f"{elem} {x:.6f} {y:.6f} {z:.6f}")
            species.append(elem)
            coords.append(site.coords)
        xyz_lines.insert(0,str(len(species)))
        logger.info(xyz_lines)

        fragment_graphs = OpenBabelToMolGraph(file_name, xyz_lines,species,coords,supercell)
        if fragment_graphs == "TimeoutException":
            filtered_mols = "TimeoutException"
            return(filtered_mols)
        else:
            all_molecules = []
            for frag in fragment_graphs:
                if fragment_type == 'coord': 
                    mol_rdkit_setting = True
                else: 
                    mol_rdkit_setting = False
                rd_mol = molgraph_to_rdkit(frag,mol_rdkit_setting)
                if rd_mol is not None:
                    all_molecules.append(rd_mol)
            filtered_mols = remove_duplicates_and_substructures(all_molecules, file_name.split('/')[-1], db_code, chemical_name_systematic, db_property,first_name)
            return(filtered_mols)
    except ValueError:
        filtered_mols = []
        return(filtered_mols)
        

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

def run(
    input_file: Path,
    ccdc_chemical_name_systematic: str = "_chemical_name_systematic",
    db_code_pattern: str = "_database_code_depnum_ccdc_archive",
    min_occ: float = 0.5,
    fragment_type: str = "coord",
    property: str = "meelting_point",
    TIMEOUT: int = 3000
    ) -> int:
    logger.info('Сервис запущен')

    if not input_file.exists():
        logger.info("[red]Папка %s не найдена:[/red]", input_file)
        return 2

    print(f"[blue]Запуск CiFragmenter в папке [bold]{input_file}[/bold][/blue]")
    
    cif_files = list(Path(input_file).glob("*.cif"))
    filtered_mols = []
    processed = 0
    input_dir = Path(input_file)
    done_dir = input_dir / "done"
    done_dir.mkdir(exist_ok=True)
    error_dir = input_dir / "error"
    error_dir.mkdir(exist_ok=True)
    mol_dir = input_dir / "mols"
    mol_dir.mkdir(exist_ok=True)
    timeout_dir = input_dir / "timeout"
    timeout_dir.mkdir(exist_ok=True)

    signal.signal(signal.SIGALRM, timeout_handler)
    num_long_time_files = 0
    er_cifs=[]
    num_ok_cifs = 0
    num_error_cifs = 0
    total = len(cif_files)
    
    cif_files = sorted(input_dir.glob("*.cif"))  
    for cif in track(cif_files, description="Фрагментация..."):
        cif_path = input_dir / cif.name
        try:
            signal.alarm(TIMEOUT)
            cif_to_mols_res = cif_to_mols(str(input_file)+'/'+cif.name, cif.name, ccdc_chemical_name_systematic, db_code_pattern, min_occ,fragment_type,property)
        except TimeoutException:
            signal.alarm(0)
            num_long_time_files += 1
            shutil.move(str(cif_path), str(timeout_dir / cif.name))
            continue
        finally:
            signal.alarm(0)
        if type(cif_to_mols_res) == str and cif_to_mols_res == "TimeoutException":
            num_long_time_files += 1
            shutil.move(str(cif_path), str(timeout_dir / cif.name))
            continue
        if len(cif_to_mols_res) == 0:
            er_cifs.append(cif.name)
            num_error_cifs += 1
            shutil.move(str(cif_path), str(error_dir / cif.name))
        else:
            num_ok_cifs += 1
            shutil.move(str(cif_path), str(done_dir / cif.name))
            for i, mol in enumerate(cif_to_mols_res["mol"], start=1):
                if mol is None:
                    continue
                w = Chem.SDWriter(str(mol_dir / f"{cif.stem}_{i}.sdf"))
                w.write(mol)
                w.close()
        processed += 1

    output_sdf = input_dir / "SDF.sdf"     
    writer = Chem.SDWriter(str(output_sdf))
    for sdf_file in sorted(mol_dir.glob("*.sdf")):
        supplier = Chem.SDMolSupplier(str(sdf_file), removeHs=False)
        for mol in supplier:
            if mol is None:
                continue
            writer.write(mol)
    writer.close()
        
    table = Table(title="Результаты")
    table.add_column("Всего cif")
    table.add_column("Успешно")
    table.add_column("Ошибка (пропущено)")
    table.add_column("Долго (пропущено)")
    table.add_column("Найдено фрагментов")
    table.add_row(str(total),str(num_ok_cifs),str(num_error_cifs),str(num_long_time_files),str(len(list(mol_dir.glob("*.sdf")))))
    table.add_row("Перемещены",str(input_file)+"/done",str(input_file)+"/error",str(input_file)+"/timeout",str(output_sdf))
    print(table)
    print("[green]Готово[/green]")
    return 0