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

All python files mentioned below should be opened in VS Code on the SCC using the virtual enviornment created above. 

Part A:

For Part A, please refer to PartA_fluid_flow.py
In this example, I explore uniform fluid flow in a pipe with the Navier Stokes equation. A pressure differential moves fluid through a pipe and no-slip boundry conditions at the edges of the pipe lead to a parabolic flow profile seen in the image below. Fenicsx was used to create the mesh and solve the partial differential equation. 

![pipe_flow_visual](https://github.com/user-attachments/assets/0a8b946a-456a-4988-805b-633e06942785)



Part B:

Refer to PartB_Beam_Bending.py. In part B, I explored mesh refinement. I used a simple fixed-free beam in this example to study how the weight of the beam would lead to a displacement at different points along the beams length. I adjusted the number of nodes along the length of the beam to see when the solution for maximum displacment converged and the additional nodes were no longer increasing the accuracy of the solution and only costing additional computing power. The first image below shows the displacement of the beam, and the second image shows the h-refinement graph. 

![deflection](https://github.com/user-attachments/assets/fc328f71-a128-429d-9158-9be509142a0d)

![Mesh Refinement](https://github.com/user-attachments/assets/a979d576-80ea-4ece-81cb-7b6f32d7f60f)



Part C:

Refer to PartC_fluid_flow.py. In Part C, I explored how Fenicsx would fail to solve a specific problem. I once again returned to the fluid flow problem explored in Part A. In this example, I adjusted the number of steps used to solve the differential equation to 5 steps from 500 steps. This reduction in number of setps lead to a strange outcome, the flow reversed in the pipe as seen in the image below. This result is clearly not correct given the boundry conditions and the pressure differential. The reduction in steps to this level lead to a false result, proving the importance of understanding results and checking answers when working with open source software. 

![PartC_pipe_flow_visual](https://github.com/user-attachments/assets/675fe48d-da3d-4b12-a6c9-8a6165b00b8b)

