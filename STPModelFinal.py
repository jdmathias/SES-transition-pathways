#######################################################################################
########################################### SE Tipping point model ####################
#######################################################################################
from IPython import get_ipython
get_ipython().magic('reset -sf')
from pylab import *
from numpy import *
from matplotlib import *
import time
import pycuda.autoinit
import pycuda.driver as drv
import random
tic=time.clock();

######################################################################################################
############################################# Start - Fonction pycuda for using GPU
from pycuda.compiler import SourceModule
mod = SourceModule("""
    #include <math.h>
    /* Opinion dynamics that depend on social interaction and perception */
__global__ void social_dynamics(float *dest1l, float *dest2l, float *a1l, float *a2l, float *u1l, float *u2l, float *rx1l, float *rx2l)
{
 const int i = blockDim.x * blockIdx.x + threadIdx.x;
 if (fabs(a1l[i]-a2l[i])>u1l[i]) // Test  according to uncertainty u1l (u in the paper) for agent 1
  {  
     dest1l[i]=a1l[i]*(1-rx1l[i]); // Opinion dynamics only depends on perception
  }
  else
  {
     dest1l[i]=a1l[i]*(1-rx1l[i])+0.5*(a2l[i]-a1l[i])*(1-rx1l[i]); // Opinion dynamics  depends on perception and social interactions
  }
 if (fabs(a1l[i]-a2l[i])>u2l[i]) // Test  according to uncertainty u2l (u in the paper) for agent 2
  {  
     dest2l[i]=a2l[i]*(1-rx2l[i]); // Opinion dynamics only depends on perception
  }
  else
  {
     dest2l[i]=a2l[i]*(1-rx2l[i])+0.5*(a1l[i]-a2l[i])*(1-rx2l[i]); // Opinion dynamics  depends on perception and social interactions
  } 
}
   /* Opinion to action according to cognitive dissonance AND perception function Pi(t)  */
__global__ void opinion_action(float *Ll, float *rl, float *Lil, float *xl, float *gxl, float ql, float Pl, float t1l)
{
 const int i = blockDim.x * blockIdx.x + threadIdx.x;
 if (fabs(Lil[i]-xl[i])>t1l) // Test for cognitive dissonance according to threshold t1l (D in the paper)  
 {  
    Ll[i]=xl[i];
 }
 else
 {
    Ll[i]=Lil[i];
 }
 rl[i]=1-pow(Pl,ql)/(pow(Pl,ql)+pow(gxl[i],ql));// Perception function Pi(t)  
}
""")
############################################# End - Fonction pycuda for using GPU
######################################################################################################
social_dynamics = mod.get_function("social_dynamics")#simplify the name of the command for GPU
opinion_action = mod.get_function("opinion_action")#simplify the name of the command for GPU

######################################################################################################
################################################### Dynamics - Main program
######################################################################################################

################################################### Parameters  ######################################
############ General parameters
temps = 250; # Total time of the simulation
nombrethreads=200;nombreblock=5; #GPU parameters

############ Population parameters
N=1000;#Size of the population
mitai=int(N/2);#Size of the half population (used for the opinion dynamics)
vect=list(range(0,N));# creation of a vector that numbers agents 

############ Opinion parameters
x = numpy.random.rand(temps+1,N).astype(numpy.float32);#Opinion vector of the population
ux = numpy.zeros_like(x);#Uncertainty vector of the population
ux[0,0:N]=ones((1,N))*0.00021;#uncertainty of moderate users
ux[0,0:10]=0;ux[0,(N-10):(N)]=0;#uncertainty of engaged users

############ Exploitation parameters
Hini=0.5;#mean of the exploitation of the population for initialization of the exploitation
sigL=0.1;#standard deviation of the exploitation of the population for initialization of the exploitation

Hx = numpy.zeros_like(x);#Initialization of the individual exploitation;
Hx[0,0:N] = (randn(1,N)*0.1+1)*Hini/N;#Exploitation of the moderate individuals;
Hx[0,(where(Hx[0,:]<0))]=0;#Check if negative values of exploitation
Hx[0,0:10]=0.3/N;#Exploitation of the ecological individuals;
Hx[0,(N-10):(N)]=0.7/N;#Exploitation of the productive individuals;
H = numpy.random.rand(temps+1,1).astype(numpy.float32);#initialization of the population exploitation
H[0,0]=numpy.sum(Hx[0,0:N])#initialization of the population exploitation

############ "Opinion to exploitation" parameter
D=0.2*Hini/N;#cognitive dissonance

############ Perception parameters
q=10;teta=0.775*1;#perception parameters.It is the refence value for Figure 4. Multiply teta by 0.00000001 in order to reproduce results of Figure 3 Multiply teta by 0.9 or 1.1 in order to reproduce results of Figure 5;
gx = numpy.zeros_like(x);#Perception vector of the population
gx[0,0:N]=teta;#Perception of the moderata
gx[0,0:10]=teta*2;#Perception of the ecological user
gx[0,(N-10):(N)]=teta/2;#Perception of the productive user
rx = numpy.zeros_like(x);#initialization of the perception of the population

############ Ecological parameters
r=0.25;K=3;#Parameters of the logistic growth of the biomass
B = numpy.zeros_like(H);#Initialization of the biomass
B[0,0]=4;#Initial value of the biomass

################################################## Initialization of the dynamical loop #####################################
############ Social initialization
x[0,0:N]=Hx[0,0:N]#at the beginning, we suppose that opinions and exploitations match
criteret=0;compttemps=0;

################################################## Start of the dynamical loop ##########################################
while compttemps<temps:         
  compttemps=compttemps+1;
  ############ Social dynamics
  random.shuffle(vect);# Population is shuffled
  a1=x[compttemps-1,vect[0:mitai]];# The population is divided in 2 parts (a1 and a2) for pair interactions
  a2=x[compttemps-1,vect[mitai:(N+1)]];# The population is divided in 2 parts (a1 and a2) for pair interactions
  u1=ux[0,vect[0:mitai]];#Uncertainties of the first part of the population a1
  u2=ux[0,vect[mitai:(N+1)]];#Uncertainties of the second part of the population a2
  rx1=rx[compttemps-1,vect[0:mitai]];#Perception of the first part of the population a1
  rx2=rx[compttemps-1,vect[mitai:(N+1)]];#Perception of the second part of the population a2
  res1 = numpy.zeros_like(a1);#initialization of an opinion vector for the results of the opinion dynamics, first part of the population a1
  res2 = numpy.zeros_like(a2);#initialization of an opinion vector for the results of the opinion dynamics, second part of the population a2
  ########### calculation of the social dynamics using GPU
  social_dynamics(
          drv.Out(res1), drv.Out(res2), drv.In(a1), drv.In(a2), drv.In(u1), drv.In(u2), drv.In(rx1), drv.In(rx2),
          block=(nombrethreads,1,1), grid=(nombreblock,1,1))
  ########### end of the calculation of the social dynamics using GPU
  x[compttemps,vect[0:mitai]]=res1;#updating of the opinion of the first part of the population a1
  x[compttemps,vect[mitai:(N+1)]]=res2;#updating of the opinion of the second part of the population a2
  if (sum(abs(a1-res1))+sum(abs(a2-res2)))<0.01:#convergence criterion: if opinions don't change, the loop stops
        criteret=1;
  ############ Ecological dynamics
  B[compttemps,0]=B[compttemps-1,0]+(-H[compttemps-1,0]+r*B[compttemps-1,0]*(K-B[compttemps-1,0]));#biomass dynamics
  if B[compttemps,0]<0:#Check if biomass value is negative because of numerical errors
      B[compttemps,0]=0;
  ############ From opinion to action/From action to opinion
  H1=Hx[compttemps-1,0:N];x1=x[compttemps,0:N];gx1=gx[0,0:N];#inputs for GPU
  resL = numpy.zeros_like(H1);#initialization of an exploitation vector for the results of the opinion_action dynamics,
  resr = numpy.zeros_like(H1);#initialization of a perception vector for the results of the opinion_action dynamics,
  ########### calculation of the opinion_action and perception dynamics using GPU
  opinion_action(
          drv.Out(resL), drv.Out(resr), drv.In(H1), drv.In(x1), drv.In(gx1), numpy.float32(q), numpy.float32(B[compttemps,0]), numpy.float32(D),
          block=(nombrethreads,1,1), grid=(nombreblock,1,1))
   ########### end of calculation of the opinion_action and perception dynamics using GPU
  Hx[compttemps,0:N]=resL;#updating of the exploitation of each individual
  rx[compttemps,0:N]=resr;#updating of the perception of each individual
  H[compttemps,0]=numpy.sum(Hx[compttemps,0:N]);#updating of the population exploitation (sum of the individual exploitations)
################################################## End of the dynamical loop ##########################################

#############################################################################################################       
################################################## Post-processing ##########################################
#############################################################################################################
toc=time.clock();
restime=toc-tic;  
print("%f" %restime)
yc = np.arange(0., K, 0.01);
xc = r*yc*(K-yc);

pyplot.plot(x[0:compttemps,0:N],'k.',x[0:compttemps,0:10],'r.',x[0:compttemps,(N-10):N],'r.');
pyplot.xlabel('Time')
pyplot.ylabel('Opinion about the exploitation level')
axes = pyplot.gca()
axes.set_ylim(0,0.00075)
axes.set_xlim(0,temps)
show()

pyplot.plot(H[0:compttemps,0],B[0:compttemps,0],xc,yc,'--')
pyplot.xlabel('Exploitation E ')
pyplot.ylabel('State x of the ecosystem (biomass)')
axes = pyplot.gca()
axes.set_ylim(0,4)
axes.set_xlim(0,0.65)
show()
        
pyplot.plot(Hx[0:compttemps,0:N],'k.',Hx[0:compttemps,0:10],'r.',Hx[0:compttemps,(N-10):N],'r.');
pyplot.xlabel('Time')
pyplot.ylabel('Exploitation level')
axes = pyplot.gca()
axes.set_ylim(0,0.00075)
axes.set_xlim(0,temps)
show()


pyplot.plot(rx[0:compttemps,10:(N-10)],'k.');
pyplot.xlabel('Time')
pyplot.ylabel('Perception')
show()
