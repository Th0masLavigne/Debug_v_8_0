# Thomas Lavigne
# 23-04-2024
#
#
# Packages
from basix.ufl import element, mixed_element
from dolfinx.io        import XDMFFile, gmshio
from dolfinx.fem       import (Constant, functionspace, Function, locate_dofs_topological, dirichletbc,
									Expression, assemble_scalar, form)
from ufl 			   import (FacetNormal, Measure, Identity, nabla_div, sym, grad, inner, 
									TestFunctions, TrialFunction,derivative,  
									split, dot,sym, sqrt,div,atan)
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py.PETSc    import ScalarType
from mpi4py            import MPI
import numpy as np
import time
#
#
##############################################################
##############################################################
##########################  Functions  #######################
##############################################################
##############################################################
# 
#
def epsilon(u):
	"""
	Compute the deformation vector
	Inputs: Displacement vector
	Outputs: Desformation vector
	"""
	return sym(grad(u))
# 
def teff(u,lambda_m,mu):
	"""
	Compute the effective Cauchy stress tensor
	Inputs: displacement vector, Lame coefficients
	Outputs: effective stress
	"""
	return lambda_m * nabla_div(u) * Identity(len(u)) + 2*mu*epsilon(u)
#
def mechanical_load(t,t_ramp,magnitude,ti,t_sustained):
	"""
	Temporal evolution function of the load
	Inputs:
	- t,t_ramp,ti: current time, duration of the ramp and starting time of the ramp in seconds
	- magnitude: magnitude in pascal 
	Outputs: 
	- Instantaneous load value
	"""
	import numpy as np
	if t<ti:
		f1=0
	elif t < (ti+t_ramp):
		tstart=ti
		# the first 0.5 is for half of the magnitude : 20mL or 40 mL
		f1 = 0.5*(0.5 * (1 - np.cos(np.pi*(t-tstart)/(t_ramp))))
	elif t < (ti+t_ramp+t_sustained):
		f1 = 0.5*1
	elif t < (ti+2*t_ramp+t_sustained):
		tstart=ti+t_ramp+t_sustained
		f1 = 0.5*(0.5 * (1 + np.cos(np.pi*(t-tstart)/(t_ramp))))
	elif t < (ti+2*t_ramp+2*t_sustained):
		f1=0
	elif t < (ti+3*t_ramp+2*t_sustained):
		tstart=ti+2*t_ramp+2*t_sustained
		f1 = 0.5*(0.5 * (1 - np.cos(np.pi*(t-tstart)/(t_ramp))))
	elif t < (ti+3*t_ramp+3*t_sustained):
		f1 = 0.5*1
	elif t < (ti+4*t_ramp+3*t_sustained):
		tstart=ti+3*t_ramp+3*t_sustained
		f1 = 0.5*(0.5 * (1 + np.cos(np.pi*(t-tstart)/(t_ramp))))
	elif t < (ti+4*t_ramp+4*t_sustained):
		f1=0
	elif t < (ti+5*t_ramp+4*t_sustained):
		tstart=ti+4*t_ramp+4*t_sustained
		f1 = (0.5 * (1 - np.cos(np.pi*(t-tstart)/(t_ramp))))
	elif t < (ti+5*t_ramp+5*t_sustained):
		f1 = 1
	elif t < (ti+6*t_ramp+5*t_sustained):
		tstart=ti+5*t_ramp+5*t_sustained
		f1 = (0.5 * (1 + np.cos(np.pi*(t-tstart)/(t_ramp))))
	elif t < (ti+6*t_ramp+6*t_sustained):
		f1=0
	elif t < (ti+7*t_ramp+6*t_sustained):
		tstart=ti+6*t_ramp+6*t_sustained
		f1 = (0.5 * (1 - np.cos(np.pi*(t-tstart)/(t_ramp))))
	elif t < (ti+7*t_ramp+7*t_sustained):
		f1 = 1
	elif t < (ti+8*t_ramp+7*t_sustained):
		tstart=ti+7*t_ramp+7*t_sustained
		f1 = (0.5 * (1 + np.cos(np.pi*(t-tstart)/(t_ramp))))
	else:
		f1=0
	return -magnitude*f1
# 
def beta(pl,pb):
	# return poro_b_0.value*(1-2*(pl-pb)/Comp_b.value)	
	return poro_b_0*(1-2*(pl-pb)/Comp_b)
#
##############################################################
##############################################################
########################## Computation #######################
##############################################################
##############################################################
# 
# Set time counter
begin_t = time.time()
#
#------------------------------------------------------------#
#                   Load the Geometry from GMSH              #
#------------------------------------------------------------#
# 
filename = "./Mesh_large.msh"
# 
mesh, cell_tag, facet_tag = gmshio.read_from_msh(filename, MPI.COMM_WORLD, 0, gdim=3)
# Kept for next trial
# Identify indices of the cells for each region for material definition
tissue_indices = [x for x in cell_tag.indices if (cell_tag.values[x] == 100)] + [x for x in cell_tag.indices if (cell_tag.values[x] == 200)]
# 
try :
	assert(len(cell_tag.indices) == len(tissue_indices))
	if MPI.COMM_WORLD.rank       == 0:
		print("All cell tags have been attributed")
except:
	if MPI.COMM_WORLD.rank       == 0:
		print("*************") 
		print("Forgotten tags => material badly defined")
		print("*************") 
		exit()
# 
# 
#
#------------------------------------------------------------#
#                   Time and Mechanical load                 #
#------------------------------------------------------------#
# 
# Initialise mechanical load
load_magnitude = 3e4 #[Pa]
# 
# BLOOD FLOW [m3/sec * 1/m2]
# BFLOW = Constant(mesh, ScalarType(4.e-6))
# PB_IMP = Constant(mesh, ScalarType(BFLOW.value * (mu_b.value/k_b_0_skin.value) * 0.006))
PB_IMP = Constant(mesh, ScalarType(50))
PB_hom = Constant(mesh, ScalarType(-PB_IMP.value))
if MPI.COMM_WORLD.rank       == 0:
	print("pression imposee= ",PB_IMP.value)
# 
T       = Constant(mesh,ScalarType(0))
ddT     = Constant(mesh,ScalarType(0))
# Initialise time constants
t           = 0
ti          = 5
t_ramp      = 20
t_sustained = 60
dt          = Constant(mesh, ScalarType(1.0))
num_steps   = int(1/dt.value*(ti+8*(t_ramp+t_sustained)))
num_steps=6
# 
#------------------------------------------------------------#
#                   Function Spaces                          #
#------------------------------------------------------------#
# Updated mesh space
updated_mesh_space    = functionspace(mesh, mesh.ufl_domain().ufl_coordinate_element())
# Parameter space
DG0_space = functionspace(mesh, ("DG", 0))
# Mixed Space (R2,R) -> (u,p)
P1_v     = element("P", mesh.topology.cell_name(), degree=1, shape=(mesh.topology.dim,))
P2       = element("P", mesh.topology.cell_name(), degree=2, shape=(mesh.topology.dim,))
P1       = element("P", mesh.topology.cell_name(), degree=1)

# 
P1v_space        = functionspace(mesh, P1_v)
P2_space 	      = functionspace(mesh, P2)
P1_space 		  = functionspace(mesh, P1)
MS                = functionspace(mesh=mesh, element=mixed_element([P2,P1,P1]))
# 
tensor_elem  = element("P", mesh.topology.cell_name(), degree=1, shape=(3,3))
tensor_space = functionspace(mesh, tensor_elem)
# 
# 
#------------------------------------------------------------#
#                   Material Definition                      #
#------------------------------------------------------------#
# 
# Solid scaffold
# Young Moduli [Pa]
E          = Constant(mesh, ScalarType(150e3)) 
# Possion's ratios [-]
nu        = Constant(mesh, ScalarType(0.48))
# Lamé Coefficients
lmbda_m	       = Constant(mesh, ScalarType(E.value*nu.value/((1+nu.value)*(1-2*nu.value))))   
mu_m           = Constant(mesh, ScalarType(E.value/(2*(1+nu.value))))  
# 
# Porous material
# Interstitial fluid viscosity [Pa.s]
viscosity      = Constant(mesh, ScalarType(5.0))   
# Interstitial porosity [-]
porosity_      = Constant(mesh, ScalarType(0.6))
# Intrinsic permeabilitty [m^2]
permeability_  = Constant(mesh, ScalarType(5e-14))
# 
# Vascular compartment
# Vessels data
alphab   = 3
#compressibility of the vessels [Pa]
Comp_b   = Constant(mesh, ScalarType(800))
#dynamic viscosity of the blood [Pa s] 		    
mu_b     = Constant(mesh, ScalarType(4e-3))
#initial porosity of vascular part []
poro_b_0_ = Constant(mesh, ScalarType(0.04))
#intrinsic permeability of vessels [m2]
k_b_0_    = Constant(mesh, ScalarType(4e-12))
# 
# Map the Lame Coefficients at the current time step
lambda_m                               = Function(DG0_space)
lambda_m.x.array[tissue_indices]       = np.full_like(tissue_indices, lmbda_m.value, dtype=ScalarType)
lambda_m.x.scatter_forward()
# 
mu                               = Function(DG0_space)
mu.x.array[tissue_indices]       = np.full_like(tissue_indices, mu_m.value, dtype=ScalarType)
mu.x.scatter_forward()
# 
# Map the porosity at the current time step
porosity                               = Function(DG0_space)
porosity.x.array[tissue_indices]       = np.full_like(tissue_indices, porosity_.value, dtype=ScalarType)
porosity.x.scatter_forward()
# 
# Map the Porosity at the previous time step
porosity_n                             = Function(DG0_space)
porosity_n.x.array[:]                  = porosity.x.array[:]
porosity_n.x.scatter_forward()
# 
# Mapping the permeabilitty
permeability                               = Function(DG0_space)
permeability.x.array[tissue_indices]       = np.full_like(tissue_indices, permeability_.value, dtype=ScalarType)
permeability.x.scatter_forward()
# 
# Map the vascular porosity
poro_b_0                             = Function(DG0_space)
poro_b_0.x.array[tissue_indices]     = np.full_like(tissue_indices, poro_b_0_.value, dtype=ScalarType)
poro_b_0.x.scatter_forward()
# 
# Mapping the permeabilitty
k_b_0                               = Function(DG0_space)
k_b_0.x.array[tissue_indices]       = np.full_like(tissue_indices, k_b_0_.value, dtype=ScalarType)
k_b_0.x.scatter_forward()
# 
#------------------------------------------------------------#
#               Functions and expressions                    #
#------------------------------------------------------------#
# 
# Integral functions
# Specify the desired quadrature degree
q_deg = 4
dx    = Measure('dx', metadata={"quadrature_degree":q_deg}, subdomain_data=cell_tag, domain=mesh)
ds    = Measure("ds", domain=mesh, subdomain_data=facet_tag)
# 
X0 = Function(MS)
Xn = Function(MS)
# 
d_u_update                = Function(updated_mesh_space)
# 
# 
# displacement increment, IF pressure increment, Blood Pressure increment
du, dpl, dpb       = split(X0)
# Previous time steps solutions
u_n, pl_n, pb_n = split(Xn)
# 
# Mapping in the Mixed Space
Un_, Un_to_MS = MS.sub(0).collapse()
Pn_, Pn_to_MS = MS.sub(1).collapse()
Pnb_, Pnb_to_MS = MS.sub(2).collapse()
# 
poro_b   = Function(DG0_space)
poro_b.name="poro_b"
k_b   = Function(DG0_space)
k_b.name="k_b"
# 
# Deformation of du (max of abs)
deformation = Function(tensor_space) 
deformation.name = "Max_def"
# 
# 
poro_b_expr = Expression(poro_b_0*(1-(2/np.pi*atan((Xn.sub(1)-Xn.sub(2))/Comp_b))), DG0_space.element.interpolation_points())
# 
k_b_expr = Expression(k_b_0*(poro_b/poro_b_0)**alphab, DG0_space.element.interpolation_points())
# 
# Update of the porosity (ask ref to S Urcun)
poro_expr         = Expression(porosity_n + (1-porosity_n)*div((X0.sub(0))), DG0_space.element.interpolation_points())
# 
# 
# Compute the solid deformation increment
deformation_du_expr = Expression(sym(grad(X0.sub(0))), tensor_space.element.interpolation_points())
# 
# 
#------------------------------------------------------------#
#            Initial and Boundary Conditions                 #
#------------------------------------------------------------#
# 
# 
poro_b.x.array[:] = poro_b_0.x.array[:]
poro_b.x.scatter_forward()
# 
k_b.x.array[:]    = k_b_0.x.array[:]
k_b.x.scatter_forward()
# 
# Initial pressures
FPn_ = Function(Pn_)
with FPn_.vector.localForm() as initial_local:
	initial_local.set(ScalarType(0)) 
Xn.x.array[Pn_to_MS] = FPn_.x.array
Xn.x.scatter_forward()
# 
FPnb_ = Function(Pnb_)
with FPnb_.vector.localForm() as initial_local:
	# beware for BCs
	initial_local.set(ScalarType(PB_IMP.value)) 
	# initial_local.set(ScalarType(0)) 
Xn.x.array[Pnb_to_MS] = FPnb_.x.array
Xn.x.scatter_forward()
# 
# 
# 
# Dirichlet BCs
# 
# 	Semi-infinite boundaries
# 1 = loading, 2 = top minus loading, 3 = bottom, 4 = left, 5 = right, 6 = Front, 7 = back
bcs    = []
fdim = mesh.topology.dim - 1
# 
facets = facet_tag.find(4)
# dux=0
dofs   = locate_dofs_topological(MS.sub(0).sub(0), fdim, facets)
bcs.append(dirichletbc(ScalarType(0), dofs, MS.sub(0).sub(0)))
# dpl = 0
dofs   = locate_dofs_topological(MS.sub(1), fdim, facets)
bcs.append(dirichletbc(ScalarType(0), dofs, MS.sub(1)))
# dPB = Pimp at t = 0 then 0
dofs   = locate_dofs_topological(MS.sub(2), fdim, facets)
bcs.append(dirichletbc(ScalarType(0.), dofs, MS.sub(2)))
# 
facets = facet_tag.find(5)
# dux=0
dofs   = locate_dofs_topological(MS.sub(0).sub(0), fdim, facets)
bcs.append(dirichletbc(ScalarType(0), dofs, MS.sub(0).sub(0)))
# dpl=0
dofs   = locate_dofs_topological(MS.sub(1), fdim, facets)
bcs.append(dirichletbc(ScalarType(0), dofs, MS.sub(1)))
# dpb=0
dofs   = locate_dofs_topological(MS.sub(2), fdim, facets)
bcs.append(dirichletbc(PB_hom, dofs, MS.sub(2)))
# 
facets = facet_tag.find(6)
# duy=0
dofs   = locate_dofs_topological(MS.sub(0).sub(1), fdim, facets)
bcs.append(dirichletbc(ScalarType(0), dofs, MS.sub(0).sub(1)))
# 
facets = facet_tag.find(7)
# duy=0
dofs   = locate_dofs_topological(MS.sub(0).sub(1), fdim, facets)
bcs.append(dirichletbc(ScalarType(0), dofs, MS.sub(0).sub(1)))
# dpl = 0
dofs   = locate_dofs_topological(MS.sub(1), fdim, facets)
bcs.append(dirichletbc(ScalarType(0), dofs, MS.sub(1)))
# duz=0
facets = facet_tag.find(3)
dofs   = locate_dofs_topological(MS.sub(0).sub(2), fdim, facets)
bcs.append(dirichletbc(ScalarType(0), dofs, MS.sub(0).sub(2)))
# 
# 
#------------------------------------------------------------#
#                     Variationnal form                      #
#------------------------------------------------------------#
#
normal = FacetNormal(mesh)
# 
v,ql,qb = TestFunctions(MS)
# 
F = (1-poro_b)*(1/(dt))*nabla_div(du)*ql*dx + ( permeability/(viscosity) )*dot( grad(pl_n+dpl),grad(ql) )*dx - (2*poro_b_0/(np.pi*Comp_b))*( (1/(dt))*((dpb-dpl)/(1+((pl_n+dpl-pb_n-dpb)/Comp_b)**2)) )*ql*dx
F += poro_b*(1/(dt))*nabla_div(du)*qb*dx + ( k_b/(mu_b) )*dot( grad(pb_n+dpb),grad(qb) )*dx + (2*poro_b_0/(np.pi*Comp_b))*( (1/(dt))*((dpb-dpl)/(1+((pl_n+dpl-pb_n-dpb)/Comp_b)**2)) )*qb*dx
F += inner(grad(v),teff(u_n+du,lambda_m,mu))*dx - (1-beta(pl_n+dpl,pb_n+dpb))*(pl_n+dpl)*nabla_div(v)*dx - beta(pl_n+dpl,pb_n+dpb)*(pb_n+dpb)*nabla_div(v)*dx - T*inner(v,normal)*ds(1)
# 
#------------------------------------------------------------#
#                           Solver                           #
#------------------------------------------------------------#
# 
# Non linear problem definition
dX0     = TrialFunction(MS)
J       = derivative(F, X0, dX0)
Problem = NonlinearProblem(F, X0, bcs = bcs, J = J)
# Debug instance
log_newton=True
if log_newton:
	from dolfinx import log
	log.set_log_level(log.LogLevel.INFO)
# 
# set up the non-linear solver
solver                       = NewtonSolver(mesh.comm, Problem)
# Absolute tolerance
solver.atol                  = 1e-10
# relative tolerance
solver.rtol                  = 1e-11
# Convergence criterion
solver.convergence_criterion = "incremental"
# Maximum iterations
solver.max_it                = 10
# 
#------------------------------------------------------------#
#                         Computation                        #
#------------------------------------------------------------#
# 
# 
for n in range(num_steps):
	if n==1:
		PB_hom.value = 0
	tprev = t
	t += dt.value
	t = round(t,2)
	# update the load
	T.value = mechanical_load(t,t_ramp,load_magnitude,ti,t_sustained)
	# Solve
	if MPI.COMM_WORLD.rank == 0:
		print('step: ',n+1,'/',num_steps,' load: ', T.value, ' Pa')
		print(n+1,'/',num_steps)
	try:
		num_its, converged = solver.solve(X0)
	except:
		if MPI.COMM_WORLD.rank == 0:
			print("*************") 
			print("Solver failed")
			print("*************") 
		break
	X0.x.scatter_forward()
	# 
	deformation.interpolate(deformation_du_expr)
	deformation.x.scatter_forward()
	if MPI.COMM_WORLD.rank == 0:
		print(f"100*max(abs(epsilon du)) = {100*max(abs(deformation.x.array[:]))} %")
		print(f"100*mean(abs(epsilon du)) = {100*np.mean(abs(deformation.x.array[:]))} %")
		print(f"100*std(abs(epsilon du)) = {100*np.std(abs(deformation.x.array[:]))} %")
	# 
	# 
	# Update previous solution
	Xn.x.array[:] += X0.x.array[:]
	Xn.x.scatter_forward()
	# 
	# Update porosity
	poro_b.interpolate(poro_b_expr)
	poro_b.x.scatter_forward()
	# 
	# Update permeabilitty
	k_b.interpolate(k_b_expr)
	k_b.x.scatter_forward()
	# Update porosity
	porosity.interpolate(poro_expr) 
	porosity.x.scatter_forward()
	# Update porosity at previous time step
	porosity_n.x.array[:]=porosity.x.array[:]
	porosity_n.x.scatter_forward()
	# 
	# 
	# mesh update
	# 
	d_u_update.interpolate(X0.sub(0))
	mesh.geometry.x[:,:mesh.geometry.dim] += d_u_update.x.array.reshape((-1, mesh.geometry.dim))
# 
