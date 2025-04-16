# Assignment4


Welcome to the Assignment 4 repo. Please follow the directions below to download and run the code on the SCC

Please begin by setting up a conda environment (mamba also works):
```bash
module load miniconda
```
```bash
mamba create -n fenicsx-env
```
Once the environment has been created, activate it:

```bash
mamba activate fenicsx-env
```
Install Dolfinx Mpich and Pyvista
```bash
mamba install -c conda-forge fenics-dolfinx mpich pyvista
```
Install Imageio Gmsh and PyYAML
```bash
pip install imageio
```
```bash
pip install gmsh
```
```bash
pip install PyYAML
```
Then open fluid_flow.py in vscode on the SCC and run the file

