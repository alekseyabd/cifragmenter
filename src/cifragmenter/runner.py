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
from rdkit.Chem import rdFMCS
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

from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path.cwd()

log = logging.getLogger("cifragmenter")
from .logging_conf import setup_logger
from .logging_conf import setup_logging
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
    elements = ['N','O','Cl','S','P','C','Se','F','Br','I','Si','B']
    #vnn = VoronoiNN(cutoff=cutoff, compute_adj_neighbors=False, tol=0.3, targets=elements)
    vnn = VoronoiNN(cutoff=cutoff, compute_adj_neighbors=False, tol=0.3)
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
            #if neighbor['poly_info']['face_dist']*2 < 3:
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
        
    
def remove_duplicates_and_substructures(mol_list, chem_uniq, fragment_type, cif_name, db_code, chemical_name_systematic, property):
    records = []
    
    for mol in mol_list:
        if mol is None:
            continue
        
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=True)
        mol.SetProp("cif_name",cif_name)
        mol.SetProp("db_code",db_code)
        mol.SetProp("compound_name",chemical_name_systematic)
        mol.SetProp("property", property)
        mol.SetProp("smiles", smiles)
        if fragment_type == 'coord':
            SA = re.sub(r"'| ", "", mol.GetProp('SA')).replace('[],','').replace('[','').replace(']','').split(',') if mol.HasProp('SA') else None
            SA_n = [round(float(num), 4) for num in list(filter(None, SA))]
            count = len(list(filter(None, SA)))
            if count>0:
                records.append({
                    'smiles': smiles,
                    'SA': SA_n,
                    'NumAtoms': mol.GetNumAtoms(),
                    '_mol': mol  
                })
        else:
            mol_no_h = Chem.RemoveHs(mol, sanitize=False)
            records.append({
                'smiles': Chem.MolToSmiles(mol_no_h, kekuleSmiles=False, canonical=True),
                'NumAtoms': mol_no_h.GetNumAtoms(),
                '_mol': mol,
                'SA': ''
            })
    if not records:
        return pd.DataFrame(columns=['cif_name', 'smiles', 'SA', 'NumAtoms'])

    df = pd.DataFrame(records)
    df = df.sort_values('NumAtoms', ascending=False, ignore_index=True)
   
    sa_strs = df['SA'].tolist()
    mols = df['_mol'].tolist()
    smiles = df['smiles'].tolist()

            

    keep = [True] * len(mols)
    for i in range(len(mols)):
        if not keep[i]:
            continue
        sa_i = sa_strs[i]
        mol_i = mols[i]
        
        for j in range(i + 1, len(mols)):
            if not keep[j]:
                continue
            mol_no_h = Chem.RemoveHs(mols[j], sanitize=False)
            new_mol = Chem.MolFromSmarts(rdFMCS.FindMCS((mol_i,mol_no_h), bondCompare=rdFMCS.BondCompare.CompareAny).smartsString)
            if fragment_type == 'mols':
                if new_mol.GetNumAtoms()==mol_no_h.GetNumAtoms():
                    keep[j] = False
            elif new_mol.GetNumAtoms()==mol_no_h.GetNumAtoms() and set(sa_strs[j]).issubset(sa_i):
                keep[j] = False
            else:
                continue
            
    df = df.loc[keep].reset_index(drop=True)

    if chem_uniq == 'chem' or fragment_type == 'mols':
        grouped = df.groupby('smiles', sort=False)
        rows_to_drop = []
        updated_indices = []

        for smiles, group in grouped:
            if len(group) <= 1:
                continue
                
            indices = group.index.tolist()
            main_idx = indices[0]      # Первая строка (оставляем её)
            dup_indices = indices[1:]  # Остальные строки (удаляем их)
            
            rows_to_drop.extend(dup_indices)
            updated_indices.append(main_idx)
            
            main_mol = df.loc[main_idx, '_mol']
            dup_mols = [df.loc[idx, '_mol'] for idx in dup_indices]
            
            prop_names = ['SA', 'Dist', 'Metals', 'PolyVolumes']
            
            for prop in prop_names:
                all_prop_values = []
                try:
                    val_str = main_mol.GetProp(prop)
                    main_list = eval(val_str) 
                except:
                    main_list = ['[]'] * 34 # Дефолтная длина, если свойства нет или ошибка парсинга
                    
                all_prop_values.append(main_list)
                
                for mol in dup_mols:
                    try:
                        val_str = mol.GetProp(prop)
                        dup_list = eval(val_str)
                    except:
                        dup_list = ['[]'] * 34
                    all_prop_values.append(dup_list)

                list_len = len(all_prop_values[0])
                
                merged_list = []
                for i in range(list_len):
                    cells = [lst[i] for lst in all_prop_values]

                    valid_values = []
                    for cell in cells:
                        if cell != '[]':
                            content = cell.strip('[]')
                            if content: # Если внутри что-то есть
                                valid_values.append(content)
                    
                    if valid_values:
                        merged_cell = str(valid_values)
                    else:
                        merged_cell = '[]'
                    
                    merged_list.append(merged_cell)
                
                new_prop_value = str(merged_list)                
                main_mol.SetProp(prop, new_prop_value)

        if rows_to_drop:
            df = df.drop(rows_to_drop).reset_index(drop=True)

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


def cif_to_mols(file_name, first_name, ccdc_chemical_name_systematic, db_code_pattern, min_occ,fragment_type,property,uniq_fragments):
    chemical_name_systematic,db_code,db_property = serach_name_ccdc(file_name,ccdc_chemical_name_systematic,db_code_pattern,property)
    try:
        CifPars = CifParser(file_name, occupancy_tolerance=6)
        struct = CifPars.get_structures()[0]
        supercell = struct.make_supercell(2,2,2)
        if fragment_type == 'coord':
            for num,i in enumerate(struct):
                
                site = str(i)
                pattern = r"\] ([a-zA-Z]+)"
                match = re.search(pattern, site)
                if(Element(match.group(1)).is_metal):
                    #if i.is_ordered == False:
                        #filtered_mols = []
                        #return(filtered_mols)
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
                        if supercell[nbr_idx].is_ordered == False:
                            filtered_mols = []
                            return(filtered_mols)
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
            filtered_mols = remove_duplicates_and_substructures(all_molecules, chem_uniq=uniq_fragments, fragment_type=fragment_type, cif_name=file_name, db_code=db_code, chemical_name_systematic=chemical_name_systematic, property=db_property)
            return(filtered_mols)
    except Exception as e:
        logger.info(f'❌ Не найдены фрагменты в {file_name}')
        logger.info(traceback.format_exc())
        filtered_mols = []
        return(filtered_mols)   

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

def process_one_cif(args):
    (
        cif_path,
        ccdc_chemical_name_systematic,
        db_code_pattern,
        min_occ,
        fragment_type,
        property,
        TIMEOUT,
        uniq_fragments,
        log_level
    ) = args

    setup_logging(log_level)

    cif_path = Path(cif_path)

    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT)

        cif_to_mols_res = cif_to_mols(
            str(cif_path),
            cif_path.name,
            ccdc_chemical_name_systematic,
            db_code_pattern,
            min_occ,
            fragment_type,
            property,
            uniq_fragments
        )

        signal.alarm(0)

        if isinstance(cif_to_mols_res, str) and cif_to_mols_res == "TimeoutException":
            return {
                "status": "timeout",
                "cif_name": cif_path.name,
                "cif_path": str(cif_path),
                "serialized_mols": [],
            }

        if len(cif_to_mols_res) == 0:
            return {
                "status": "error",
                "cif_name": cif_path.name,
                "cif_path": str(cif_path),
                "serialized_mols": [],
            }

        serialized_mols = []
        for mol in cif_to_mols_res["_mol"]:
            if mol is None:
                continue
            
            try:
                rdDetermineBonds.DetermineBondOrders(mol)
                Chem.SanitizeMol(mol)
            except:
                #Chem.SanitizeMol(mol)
                logger.info('Санитизация не прошла')


            mol_block = Chem.MolToMolBlock(mol)
            
            props = {}
            for prop_name in mol.GetPropNames(includePrivate=True):
                props[prop_name] = mol.GetProp(prop_name)

            serialized_mols.append({
                "mol_block": mol_block,
                "props": props,
            })

        return {
            "status": "ok",
            "cif_name": cif_path.name,
            "cif_path": str(cif_path),
            "serialized_mols": serialized_mols,
        }

    except TimeoutException:
        signal.alarm(0)
        return {
            "status": "timeout",
            "cif_name": cif_path.name,
            "cif_path": str(cif_path),
            "serialized_mols": [],
        }

    except Exception:
        signal.alarm(0)
        logger.info(traceback.format_exc())
        return {
            "status": "error",
            "cif_name": cif_path.name,
            "cif_path": str(cif_path),
            "serialized_mols": [],
        }

def run(
    input_file: Path,
    ccdc_chemical_name_systematic: str = "_chemical_name_systematic",
    db_code_pattern: str = "_database_code_depnum_ccdc_archive",
    min_occ: float = 0.5,
    fragment_type: str = "coord",
    property: str = "meelting_point",
    TIMEOUT: int = 3000,
    n_jobs: int | None = None,
    uniq_fragments: str = "chem",
    log_level: str = "INFO"
    ) -> int:
    logger.info('Сервис запущен')

    if not input_file.exists():
        logger.info("[red]Папка %s не найдена:[/red]", input_file)
        return 2

    print(f"[blue]Запуск CiFragmenter в папке [bold]{input_file}[/bold][/blue]")

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

    num_long_time_files = 0
    er_cifs = []
    num_ok_cifs = 0
    num_error_cifs = 0

    cif_files = sorted(input_dir.glob("*.cif"))
    total = len(cif_files)

    if n_jobs is None:
        cpu_count = os.cpu_count() or 1
        n_jobs = max(1, cpu_count - 1)

    tasks = [
        (
            str(cif),
            ccdc_chemical_name_systematic,
            db_code_pattern,
            min_occ,
            fragment_type,
            property,
            TIMEOUT,
            uniq_fragments,
            log_level
        )
        for cif in cif_files
    ]

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_cif = {
            executor.submit(process_one_cif, task): Path(task[0])
            for task in tasks
        }

        for future in track(
            as_completed(future_to_cif),
            total=len(future_to_cif),
            description="Фрагментация..."
        ):
            cif_path = future_to_cif[future]
            cif_name = cif_path.name

            try:
                result = future.result()
            except Exception:
                logger.info(traceback.format_exc())
                num_error_cifs += 1
                er_cifs.append(cif_name)
                if cif_path.exists():
                    shutil.move(str(cif_path), str(error_dir / cif_name))
                processed += 1
                continue

            if result["status"] == "timeout":
                num_long_time_files += 1
                if cif_path.exists():
                    shutil.move(str(cif_path), str(timeout_dir / cif_name))
                processed += 1
                continue

            if result["status"] == "error":
                logger.info(traceback.format_exc())
                er_cifs.append(cif_name)
                num_error_cifs += 1
                if cif_path.exists():
                    shutil.move(str(cif_path), str(error_dir / cif_name))
                processed += 1
                continue

            num_ok_cifs += 1
            if cif_path.exists():
                shutil.move(str(cif_path), str(done_dir / cif_name))

            for i, item in enumerate(result["serialized_mols"], start=1):
                mol = Chem.MolFromMolBlock(item["mol_block"], removeHs=False)
                if mol is None:
                    continue

                for prop_name, prop_value in item["props"].items():
                    mol.SetProp(prop_name, str(prop_value))

                w = Chem.SDWriter(str(mol_dir / f"{cif_path.stem}_{i}.sdf"))
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

    table.add_row(
        str(total),
        str(num_ok_cifs),
        str(num_error_cifs),
        str(num_long_time_files),
        str(len(list(mol_dir.glob("*.sdf"))))
    )
    table.add_row(
        "Перемещены",
        str(input_file) + "/done",
        str(input_file) + "/error",
        str(input_file) + "/timeout",
        str(output_sdf)
    )

    print(table)
    print("[green]Готово[/green]")
    return 0