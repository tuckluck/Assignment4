# Scaled variable
import pyvista
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1
W = 0.2
mu = 1
rho = 1
delta = W / L
gamma = 0.4 * delta**2
beta = 1.25
lambda_ = beta
g = gamma

displace_list = []
mesh_list = []

for j in range(2,30):
    # Create mesh and function space
    domain = mesh.create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([L, W, W])],[j, 6, 6], cell_type=mesh.CellType.hexahedron)
    V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))

    # Boundary condition
    def clamped_boundary(x):
        return np.isclose(x[0], 0)

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)
    u_D = np.array([0, 0, 0], dtype=default_scalar_type)
    bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)

    # Define forces and measures
    T = fem.Constant(domain, default_scalar_type((0, 0, 0)))
    ds = ufl.Measure("ds", domain=domain)

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(domain, default_scalar_type((0, 0, -rho * g)))
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L_form = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds

    problem = LinearProblem(a, L_form, bcs=[bc],petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    # Compute magnitude of displacement at each node
    u_array = uh.x.array.reshape((-1, 3))  # reshape to N x 3 (vector)
    u_magnitude = np.linalg.norm(u_array, axis=1)

    # Find max deflection and corresponding point index
    max_deflection = np.max(u_magnitude)
    max_index = np.argmax(u_magnitude)
    max_point_coords = domain.geometry.x[max_index]
    
    displace_list.append(max_deflection)
    mesh_list.append(j)
    
#rint(displace_list)
#print(mesh_list)


plt.plot(mesh_list,displace_list)
plt.title("H-Refinement")                 # Title of the plot
plt.xlabel("Number of Nodes Along Length")         # Label for x-axis
plt.ylabel("Max Displacement")  
plt.savefig("Beam_h-refine", dpi=300)