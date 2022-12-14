# MTF072 Computational Fluid Dynamics
# Task 3: laminar lid-driven cavity
# Template prepared by:
# Gonzalo Montero Villar
# Department of Mechanics and Maritime Sciences
# Division of Fluid Dynamics
# December 2020

#==============Packages needed=================
import matplotlib.pyplot as plt
import numpy as np

#================= Inputs =====================

# Fluid properties and B. C. inputs

UWall = 1 # velocity of the upper wall
rho   = 1 # density
nu    = 1/1000 # kinematic viscosity

data_file = 'data_FOU_CD.txt' #data file where the given solution is stored

# Geometric inputs (fixed so that a fair comparison can be made)

mI = 11 # number of mesh points X direction. 
mJ = 11 # number of mesh points Y direction. 
xL =  1 # length in X direction
yL =  1 # length in Y direction

# Solver inputs

nIterations           =  1000 # maximum number of iterations
n_inner_iterations_gs =  20 # amount of inner iterations when solving 
                              # pressure correction with Gauss-Seidel
n_inner_iterations_momentum=1 # amount of inner iterations when solving 
                              # momentum with Gauss-Seidel

resTolerance =  0.0001 # convergence criteria for residuals
                     # each variable
alphaUV =   0.4    # under relaxation factor for U and V
alphaP  =   0.7   # under relaxation factor for P

# ================ Code =======================

# For all the matrices the first input makes reference to the x coordinate
# and the second input to the y coordinate, (i+1) is east and (j+1) north

# Allocate all needed variables
nI = mI + 1                      # number of nodes in the X direction. nodes 
                                  # added in the boundaries
nJ = mJ + 1                      # number of nodes in the Y direction. nodes 
                                  # added in the boundaries
coeffsUV   = np.zeros((nI,nJ,5)) # coefficients for the U and V equation
                                  # E, W, N, S and P
sourceUV   = np.zeros((nI,nJ,2)) # source coefficients for the U and V equation
                                  # U and V
coeffsPp   = np.zeros((nI,nJ,5)) # coefficients for the pressure correction
                                  # equation E, W, N, S and P
sourcePp   = np.zeros((nI,nJ))   # source coefficients for the pressure
                                  # correction equation
U          = np.zeros((nI,nJ))   # U velocity matrix
V          = np.zeros((nI,nJ))   # V velocity matrix
P          = np.zeros((nI,nJ))   # pressure matrix
Pp         = np.zeros((nI,nJ))   # pressure correction matrix
D          = np.zeros((nI,nJ,4))
F          = np.zeros((nI,nJ,4))

massFlows  = np.zeros((nI,nJ,4)) # mass flows at the faces
                                  # m_e, m_w, m_n and m_s

residuals  = np.zeros((3,1))     # U, V and conitnuity residuals

# Generate mesh and compute geometric variables

# Allocate all variables matrices
xCoords_N = np.zeros((nI,nJ)) # X coords of the nodes
yCoords_N = np.zeros((nI,nJ)) # Y coords of the nodes
xCoords_M = np.zeros((mI,mJ)) # X coords of the mesh points
yCoords_M = np.zeros((mI,mJ)) # Y coords of the mesh points
dxe_N     = np.zeros((nI,nJ)) # X distance to east node
dxw_N     = np.zeros((nI,nJ)) # X distance to west node
dyn_N     = np.zeros((nI,nJ)) # Y distance to north node
dys_N     = np.zeros((nI,nJ)) # Y distance to south node
dx_CV      = np.zeros((nI,nJ)) # X size of the node
dy_CV      = np.zeros((nI,nJ)) # Y size of the node

residuals_U = []
residuals_V = []
residuals_c = []

dx = xL/(mI - 1)
dy = yL/(mJ - 1)

# Fill the coordinates
for i in range(mI):
    for j in range(mJ):
        # For the mesh points
        xCoords_M[i,j] = i*dx
        yCoords_M[i,j] = j*dy

        # For the nodes
        if i > 0:
            xCoords_N[i,j] = 0.5*(xCoords_M[i,j] + xCoords_M[i-1,j])
        if i == mI-1 and j>0:
            yCoords_N[i+1,j] = 0.5*(yCoords_M[i,j] + yCoords_M[i,j-1])
        if j > 0:
            yCoords_N[i,j] = 0.5*(yCoords_M[i,j] + yCoords_M[i,j-1])
        if j == mJ-1 and i>0:
            xCoords_N[i,j+1] = 0.5*(xCoords_M[i,j] + xCoords_M[i-1,j])

        # Fill dx_CV and dy_CV
        if i > 0:
            dx_CV[i,j] = xCoords_M[i,j] - xCoords_M[i-1,j]
        if j > 0:
            dy_CV[i,j] = yCoords_M[i,j] - yCoords_M[i,j-1]

xCoords_N[-1,:] = xL
yCoords_N[:,-1] = yL


# Fill dxe, dxw, dyn and dys
for i in range(1,nI-1):
    for j in range(1,nJ-1):
        dxe_N[i,j] = xCoords_N[i+1,j] - xCoords_N[i,j]
        dxw_N[i,j] = xCoords_N[i,j] - xCoords_N[i-1,j]
        dyn_N[i,j] = yCoords_N[i,j+1] - yCoords_N[i,j]
        dys_N[i,j] = yCoords_N[i,j] - yCoords_N[i,j-1]


# Initialize face velocity matrices
U_e = np.zeros((nI,nJ))
U_w = np.zeros((nI,nJ))
V_n = np.zeros((nI,nJ))
V_s = np.zeros((nI,nJ))

#Set D values
for i in range(1,nI-1):
    for j in range(1,nJ-1):
        D[i,j,0] =  nu * dy_CV[i,j] / dxe_N[i,j] # east diffusive
        D[i,j,1] =  nu * dy_CV[i,j] / dxw_N[i,j] # west diffusive
        D[i,j,2] =  nu * dx_CV[i,j] / dyn_N[i,j] # north diffusive
        D[i,j,3] =  nu * dx_CV[i,j] / dys_N[i,j] # south diffusive        



#Dirichlet conditions
# for i in range(0,nI):
#     j = nJ-1
#     U[i,j] = UWall

# Looping

for iter in range(nIterations):
    coeffsUV[:, :, :] = 0
    sourceUV[:, :, :] = 0
    sourcePp[:, :] = 0
    coeffsPp[:, :, :] = 0
    # Impose boundary conditions for velocities, only the top boundary wall
    # is moving from left to right with UWall
    coeffsUV[:,:,:] = 0
    coeffsPp[:,:,:] = 0
    sourceUV[:,:,:] = 0
    sourcePp[:,:] = 0
    U[:, -1] = UWall

    #TODO Calc Fs here
    for i in range(1, nI - 1):
        for j in range(1, nJ - 1):
            F[i,j,0] =  rho * U_e[i,j] * dy_CV[i,j]  # east convective
            F[i,j,1] =  rho * U_w[i,j] * dy_CV[i,j]  # weast convective
            F[i,j,2] =  rho * V_n[i,j] * dx_CV[i,j]  # north convective
            F[i,j,3] =  rho * V_s[i,j] * dx_CV[i,j]  # south convective
    
    # Impose pressure boundary condition, all homogeneous Neumann
    for i in range(0, nI):
        j = 0
        P[i,j] = P[i,j+1]

        j= nJ-1
        P[i,j] = P[i,j-1]

    for j in range(0,nJ):
        i = 0
        P[i,j] = P[i+1,j]

        i = nI - 1
        P[i,j] = P[i-1, j]   
    
    ## Compute coefficients for nodes
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            coeffsUV[i,j,0] = D[i,j,0] + max(0,-F[i,j,0])#ae
            coeffsUV[i,j,1] = D[i,j,1] + max(0,F[i,j,1]) #aw
            coeffsUV[i,j,2] = D[i,j,2] + max(0,-F[i,j,2])#an
            coeffsUV[i,j,3] = D[i,j,3] + max(0,F[i,j,3]) #as

            #TODO make Dirichlet bc-Coeffs
            # if i == 1:
            #     coeffsUV[i,j,1] = 0

            # if i == nI-2:
            #     coeffsUV[i,j,0] = 0

            # if j == 1:
            #     coeffsUV[i,j,3] = 0

            # if j == nJ-2:
            #     coeffsUV[i,j,2] = 0
                
            #     sourceUV[i,j,0] = sourceUV[i,j,0] + 2 * nu * dx_CV[i,j] / dy_CV[i,j] * UWall

            #Maybe remove Sp, probably not
            Sp= rho * dy_CV[i,j]* (U_w[i,j] - U_e[i,j]) + rho * dx_CV[i,j]* (V_s[i,j] - V_n[i,j])
            coeffsUV[i,j,4] = np.sum(coeffsUV[i,j,0:4])  -Sp #ap
            ## Introduce pressure source and implicit under-relaxation for U and V
            sourceUV[i,j,0] =  1/2 * (P[i-1,j] - P[i+1,j]) * dy_CV[i,j] + (1-alphaUV) * U[i,j] * coeffsUV[i,j,4] / alphaUV 
            sourceUV[i,j,1] =  1/2 * (P[i,j-1] - P[i,j+1]) * dx_CV[i,j] + (1-alphaUV) * V[i,j] * coeffsUV[i,j,4] / alphaUV 

    #Solve U, V fields, along with implicit under-relaxation factor to a_p
    for iter_gs in range(n_inner_iterations_momentum):
        for j in range(1,nJ-1):
            for i in range(1,nI-1):
                
                #U Eq
                RHS_U = coeffsUV[i,j,0] * U[i+1,j] + coeffsUV[i,j,1] * U[i-1,j] \
                	+ coeffsUV[i,j,2] * U[i,j+1] + coeffsUV[i,j,3] * U[i,j-1] + sourceUV[i,j,0]
                U[i,j] = (alphaUV * RHS_U) / coeffsUV[i,j,4]
                
                #V Eq 
                RHS_V = coeffsUV[i,j,0] * V[i+1,j] + coeffsUV[i,j,1] * V[i-1,j] \
                	+ coeffsUV[i,j,2] * V[i,j+1] + coeffsUV[i,j,3] * V[i,j-1] + sourceUV[i,j,1]
                V[i,j] = (alphaUV * RHS_V) / coeffsUV[i,j,4]

    ## Rhie Chow for velocities at faces 
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            if i != nI-2:
                a_P_e = 1/2 *(coeffsUV[i,j,4] + coeffsUV[i+1,j,4]) / alphaUV     ## divided by alphaUV here 
                U_e[i,j] = 0.5*(U[i+1,j] + U[i,j]) + ((dy_CV[i,j] / (4*a_P_e))*((P[i+2,j] - 3*P[i+1,j] + 3*P[i,j] - P[i-1,j])))

            if i != 1:
                a_P_w = 1/2 *(coeffsUV[i,j,4] + coeffsUV[i-1,j,4]) / alphaUV
                U_w[i,j] = 0.5*(U[i,j] + U[i-1,j]) + ((dy_CV[i,j] / (4*a_P_w))*((P[i+1,j] - 3*P[i,j] + 3*P[i-1,j] - P[i-2,j])))

            if j != nJ-2:
                a_P_n = 1/2 *(coeffsUV[i,j,4] + coeffsUV[i,j+1,4]) / alphaUV
                V_n[i,j] = 0.5*(V[i,j+1] + V[i,j]) + ((dx_CV[i,j] / (4*a_P_n))*((P[i,j+2] - 3*P[i,j+1] + 3*P[i,j] - P[i,j-1])))

            if j != 1:
                a_P_s = 1/2 *(coeffsUV[i,j,4] + coeffsUV[i,j-1,4]) / alphaUV
                V_s[i,j] = 0.5*(V[i,j] + V[i,j-1]) + ((dx_CV[i,j] / (4*a_P_s))*((P[i,j+1] - 3*P[i,j] + 3*P[i,j-1] - P[i,j-2])))


    ## Calculate pressure correction equation coefficients
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            # hint: set homogeneous Neumann coefficients with if 
            #Equidistand mesh
            if(i != nI-2):
                coeffsPp[i,j,0] = rho * 2 * dy_CV[i,j]**2 * alphaUV / (coeffsUV[i+1,j,4] + coeffsUV[i,j,4]) #E  ###multiplied with alphaUV in here!!
            
            if(i != 1):
                coeffsPp[i,j,1] = rho * 2 * dy_CV[i,j]**2 * alphaUV / (coeffsUV[i-1,j,4] + coeffsUV[i,j,4]) #W
            
            if(j != nJ-2):
                coeffsPp[i,j,2] = rho * 2 * dx_CV[i,j]**2 * alphaUV / (coeffsUV[i,j+1,4] + coeffsUV[i,j,4]) #N
            
            if(j != 1):
                coeffsPp[i,j,3] = rho * 2 * dx_CV[i,j]**2 * alphaUV / (coeffsUV[i,j-1,4] + coeffsUV[i,j,4]) #S
            
            coeffsPp[i,j,4] = np.sum(coeffsPp[i,j,0:4]) #P
            sourcePp[i,j]   = rho * dy_CV[i,j]* (U_w[i,j] - U_e[i,j]) + rho * dx_CV[i,j]* (V_s[i,j] - V_n[i,j]) #S_U
    
    # Solve for pressure correction (Note that more that one loop is used)
    Pp[:,:] = 0 #Reset due to stability
    for iter_gs in range(n_inner_iterations_gs):
        for j in range(1,nJ-1):
            for i in range(1,nI-1):    
                RHS_P = coeffsPp[i,j,0]*Pp[i+1,j] + \
                        coeffsPp[i,j,1] * Pp[i-1,j] \
                       + coeffsPp[i,j,2] * Pp[i,j+1] + coeffsPp[i,j,3]*Pp[i,j-1] + sourcePp[i,j]
                
                Pp[i,j] = RHS_P / coeffsPp[i,j,4]        
    
    # Set Pp with reference to node (2,2) and copy to boundaries
    Pp=Pp-Pp[1,1]

    for i in range(1, nI-1):
        j = 0
        Pp[i,j] = Pp[i,j+1]

        j = nJ-1
        Pp[i,j] = Pp[i,j-1]
    
    for j in range(1, nJ-1):
        i = 0
        Pp[i,j] = Pp[i+1, j]

        i = nI-1
        Pp[i,j] = Pp[i-1,j]
           
    # Correct velocities, pressure and mass flows
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            P[i,j] = P[i,j] + alphaP * Pp[i,j]
            dU = dy_CV[i,j] / coeffsUV[i,j,4] *alphaUV 
            U[i,j] = U[i,j] + dU * (Pp[i-1,j] - Pp[i+1,j])/2
            dV = dx_CV[i,j] / coeffsUV[i,j,4] * alphaUV
            V[i,j] = V[i,j] + dU * (Pp[i,j-1] - Pp[i,j+1])/2

    #Rhie-Chow
    
            
    #TODO impose zero mass flow at the boundaries
    # for i in range(1, nI-1):
    #     j = 0
    #     F[i,j,3] = 0

    #     j = nJ - 1
    #     F[i,j,2] = 0

    # for j in range(1, nJ-1):
    #     i = 0
    #     F[i,j,0] = 0

    #     i = nI-1
    #     F[i,j,1] = 0     
    # Compute residuals
    
    R_U = 0
    R_V = 0
    R_C = 0
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            R_U =  R_U + (1/UWall*xL) * abs(coeffsUV[i,j,4]*U[i,j] / alphaUV - coeffsUV[i,j,0]*U[i+1,j] \
                - coeffsUV[i,j,1]*U[i-1,j] - coeffsUV[i,j,2]*U[i,j+1] - coeffsUV[i,j,3]*U[i,j-1] - sourceUV[i,j,0])  
            
            R_V = R_V + (1/UWall*yL) * abs(coeffsUV[i,j,4]*V[i,j] / alphaUV - coeffsUV[i,j,0]*V[i+1,j] \
                - coeffsUV[i,j,1]*V[i-1,j] - coeffsUV[i,j,2]*V[i,j+1] - coeffsUV[i,j,3]*V[i,j-1] - sourceUV[i,j,1])
            
            R_C = R_C + abs(sourcePp[i,j])
            
            #residuals_c[-1] = 

    residuals_U.append(R_U) # U momentum residual
    residuals_V.append(R_V) # V momentum residual
    residuals_c.append(R_C) # continuity residual

    print('iteration: %d\nresU = %.5e, resV = %.5e, resCon = %.5e\n\n'\
        % (iter, residuals_U[-1], residuals_V[-1], residuals_c[-1]))
    
    #  Check convergence
    if resTolerance>max([residuals_U[-1], residuals_V[-1], residuals_c[-1]]):
        break

# Plotting section (these are some examples, more plots might be needed)


# Plot mesh
plt.figure()
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Computational mesh')

# Plot results

plt.figure()

# U velocity contour
plt.subplot(2,3,1)
plt.contourf(xCoords_N, yCoords_N, U, 50)
plt.title('U velocity [m/s]')
plt.xlabel('x [m]')
plt.ylabel('y [m]')

# V velocity contour
plt.subplot(2,3,2)
plt.title('V velocity [m/s]')
plt.contourf(xCoords_N, yCoords_N, V, 50)
plt.xlabel('x [m]')
plt.ylabel('y [m]')

# P contour
plt.subplot(2,3,3)
plt.title('Pressure [Pa]')
plt.contourf(xCoords_N, yCoords_N, P, 50)
plt.xlabel('x [m]')
plt.ylabel('y [m]')

# Vector plot
plt.subplot(2,3,4)
plt.title('Vector plot of the velocity field')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.quiver(xCoords_N, yCoords_N, U, V)

# Comparison with data
data=np.genfromtxt(data_file, skip_header=1)
uInterp = np.zeros((nJ-2,1))
vInterp = np.zeros((nJ-2,1))
for j in range(1,nJ-1):
    for i in range(1,nI-1):
        if xCoords_N[i,j]<0.5 and xCoords_N[i+1,j]>0.5:
            uInterp[j-1] = (U[i+1,j] + U[i,j])*0.5
            vInterp[j-1] = (V[i+1,j] + V[i,j])*0.5
            break
        elif abs(xCoords_N[i,j]-0.5) < 0.000001:
            uInterp[j-1] = U[i,j]
            vInterp[j-1] = V[i,j]
            break

plt.subplot(2,3,5)
plt.plot(data[:,0],data[:,2],'r.',markersize=20,label='data U')
plt.plot(data[:,1],data[:,2],'b.',markersize=20,label='data V')
plt.plot(uInterp,yCoords_N[1,1:-1],'k',label='sol U')
plt.plot(vInterp,yCoords_N[1,1:-1],'g',label='sol V')
plt.title('Comparison with data at x = 0.5')
plt.xlabel('u, v [m/s]')
plt.ylabel('y [m]')
plt.legend()

plt.subplot(2,3,6)
plt.title('Residual convergence')
plt.xlabel('iterations')
plt.ylabel('residuals [-]')
#plt.legend('U momentum','V momentum', 'Continuity')
plt.title('Residuals')
plt.show()



