import logging
from collections import OrderedDict


import lal
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from tqdm import tqdm
import lal
from astropy import units as u
from ml4gw.distributions import Cosine, Sine
from bilby.gw.conversion import transform_precessing_spins, bilby_to_lalsimulation_spins


class Constant:

    def __init__(self, val, tensor=True):
        self.val = val
        self.tensor = tensor

    def __repr__(self, cls):
        return self.__name__

    def sample(self, batch_size):

        if self.tensor:
            return torch.full(batch_size, self.val)

        return self.val

def mass12_from_chirpq_prior(
    Mc_prior,
    q_prior,
    batch_size: int = 64,
    mass_limit: list[int, int] = [5, 100]
):
    
    m1, m2 = np.zeros((batch_size,)), np.zeros((batch_size,))
    
    # Change this to hierarchical loop to boast the speed
    for i in range(batch_size):
        while(m1[i] < mass_limit[0] or m1[i] > mass_limit[1] or m2[i] < mass_limit[0] or m2[i] > mass_limit[1]):
            Mc = Mc_prior.sample((1,))
            q = q_prior.sample((1,))
            
            m1[i] = (Mc * (1.0 + q) ** 0.2 / q**0.6)[0]
            m2[i] = (m1[i] * q)[0]
            
            # m2[i] = (Mc*(1+q)**(1/5)*q**(2/5))[0]
            # m1[i] = (Mc*(1+q)**(1/5)*q**(-3/5))[0]
    
    return Mc, q, torch.Tensor(m1), torch.Tensor(m2)

class BasePrior:

    def __init__(self):
        self.params = OrderedDict()
        self.sampled_params = OrderedDict()

    def sample(self, batch_size):

        self.sampled_params = OrderedDict()
        
        for k in self.params.keys():

            if type(self.params[k]) == Constant:
                    self.sampled_params[k] = self.params[k].val
            else:
                self.sampled_params[k] = self.params[k].sample((batch_size,))

        return self.sampled_params


class SineGaussianHighFrequency(BasePrior):

    def __init__(self):
    # something with sample method that returns dict that maps
    # parameter name to tensor of parameter names
        super().__init__()

        self.params = OrderedDict(
            hrss = Uniform(1e-21, 2e-21),
            quality = Uniform(5, 75),
            frequency = Uniform(512, 1024),
            phase = Uniform(0, 2 * torch.pi),
            eccentricity = Uniform(0, 0.01),
            ra = Uniform(0, 2 * torch.pi),
            dec = Cosine(),
            psi = Uniform(0, 2 * torch.pi)
        )


class SineGaussianLowFrequency(BasePrior):

    def __init__(self):
    # something with sample method that returns dict that maps
    # parameter name to tensor of parameter names
        super().__init__()
        self.params = OrderedDict(
            hrss = Uniform(1e-21, 2e-21),
            quality = Uniform(5, 75),
            frequency = Uniform(64, 512),
            phase = Uniform(0, 2 * torch.pi),
            eccentricity = Uniform(0, 0.01),
            ra = Uniform(0, 2 * torch.pi),
            dec = Cosine(),
            psi = Uniform(0, 2 * torch.pi)
        )

class LAL_BBHPrior(BasePrior):
    
    def __init__(
        self,
        f_min=30,
        f_max=2048,
        duration=2, # duration of the time series
        f_ref=20.0
    ):

        self.priors = {}
        self.wave_params = {}
        self.sampled_params = {}
        
        self.lal_keys = [
            "incl", # Transformed inclination angle. (TensorType)
            "s1x", # Spin component x of the first BH. (TensorType)
            "s1y", # Spin component y of the first BH. (TensorType)
            "s1z", # Spin component z of the first BH. (TensorType)
            "s2x", # Spin component x of the second BH. (TensorType)
            "s2y", # Spin component y of the second BH. (TensorType)
            "s2z", # Spin component z of the second BH. (TensorType)
        ]

        # Frequency series in Hz. (TensorType)
        self.sampled_params["fs"] = torch.arange(f_min, f_max, 1 / duration) 
        
        # Chirp mass in solar masses. (TensorType)
        self.priors['chirp_mass'] = Uniform(15, 30) 
        
        # Mass ratio m1/m2. (TensorType)
        self.priors['mass_ratio'] = Uniform(0.5, 0.99) 
        
        # Luminosity distance in Mpc.(TensorType)
        self.priors["dist_mpc"] = Uniform(50, 200) 
        
        # Coalescence time. (TensorType)
        self.priors["tc"] = Constant(0) 
        
        # Sky location: Right ascension angle 
        self.priors['ra'] = Uniform(0, 2*np.pi)
        
        # Sky location: Declination angle
        self.priors['dec'] = Cosine(-np.pi/2, np.pi/2)
        
        # Phase of the two polarlization
        self.priors['psi'] = Uniform(0, 2*np.pi)
        
        # ----- Spin & incl parameters (Bilby parameters) -----
        # Inclination in bibly setup
        self.priors['theta_jn'] = Sine() 
        
        # Spin phase angle
        self.priors['phi_jl'] = Uniform(0, 2 * np.pi) 
        
        # Primary object tilt
        self.priors['tilt_1'] = Sine(0, np.pi) 
        
        # Secondary object tilt
        self.priors['tilt_2'] = Sine(0, np.pi) 
        
        # Relative spin azimuthal angle
        self.priors['phi_12'] = Uniform(0, 2 * np.pi) 
        
        # Primary dimensionless spin magnitude
        self.priors['a_1'] = Uniform(0, 0.5) 
        
        # Secondary dimensionless spin magnitude
        self.priors['a_2'] = Uniform(0, 0.5) 
        
        # Reference frequency in Hz. *****(float)*****
        self.sampled_params["f_ref"] = Constant(f_ref).sample((1,)).numpy() 
        
        # Uniform(0, 2*np.pi) # Reference phase. (TensorType) #(Bilby) Orbital phase
        self.priors['phiRef'] = Constant(0) 

        self.sample_keys = self.priors.keys()
        
    def translator(self):
        
        for key in self.priors.keys():
            
            self.wave_params[key] = self.priors[key].sample((1,)) 

        Mc = self.wave_params['chirp_mass']
        q = self.wave_params['mass_ratio']
        
        m1 = (Mc * (1.0 + q) ** 0.2 / q**0.6)[0]
        m2 = (m1 * q)[0]
        
        # This function can't do batch translation
        # would have to be torchify if it's draging down the speed
        lal_spins = bilby_to_lalsimulation_spins(
            theta_jn=self.wave_params['theta_jn'], 
            phi_jl=self.wave_params['phi_jl'],
            tilt_1=self.wave_params['tilt_1'],
            tilt_2=self.wave_params['tilt_2'],
            phi_12=self.wave_params['phi_12'],
            a_1=self.wave_params['a_1'],
            a_2=self.wave_params['a_2'],
            mass_1=torch.multiply(m1, lal.MSUN_SI).numpy(),
            mass_2=torch.multiply(m2, lal.MSUN_SI).numpy(),
            reference_frequency=self.sampled_params['f_ref'],
            phase=self.wave_params['phiRef'],
        )
        
        for i, key in enumerate(self.lal_keys):
            # breakpoint()
            # self.wave_params[key] = self.to_tensor(lal_spins[i])
            self.wave_params[key] = torch.Tensor(lal_spins[i])
        
        return self.wave_params

    def sample(self, batch_size):

        for i, key in enumerate(self.sample_keys):
            
            self.sampled_params[key] = torch.zeros((batch_size,))
            
        for i, key in enumerate(self.lal_keys):
            
            self.sampled_params[key] = torch.zeros((batch_size,))

        for i in tqdm(range(batch_size)):
            data = self.translator()

            for _, key in enumerate(self.sample_keys):

                self.sampled_params[key][i] = data[key][0]
                
            for _, key in enumerate(self.lal_keys):

                self.sampled_params[key][i] = data[key][0]

        return self.sampled_params

class BBHPrior(BasePrior):

    def __init__(self):
    # something with sample method that returns dict that maps
    # parameter name to tensor of parameter names
        super().__init__()
        # taken from bilby.gw.prior.BBHPriorDict()
        self.params = dict(
            mass_ratio = Uniform(0.5, 0.99), # Uniform(0.125, 1),
            chirp_mass = Uniform(15, 30), # Uniform(25, 100),
            theta_jn = Sine(),
            phase = Constant(0), # Uniform(0, 2 * torch.pi),
            a_1 = Uniform(0, 0.99),
            a_2 = Uniform(0, 0.99),
            tilt_1 = Sine(0, torch.pi),
            tilt_2 = Sine(0, torch.pi),
            phi_12 = Uniform(0, 2 * torch.pi),
            phi_jl = Uniform(0, 2 * torch.pi),
            reference_frequency = Constant(20.0, tensor=False), #Constant(50.0, tensor=False),
            # CHECK THIS: time of coallesence and fs
            tc = Constant(0),
            fs = Constant(2048),
            dist_mpc = Uniform(50, 200),
            ra = Uniform(0, 2 * torch.pi),
            dec = Cosine(),
            psi = Uniform(0, torch.pi)
        )

    def sample(self, batch_size):

        for k in self.params.keys():
            self.sampled_params[k] = self.params[k].sample((batch_size,))

        self.sampled_params['mass_2'] = self.sampled_params['chirp_mass'] * (1 + self.sampled_params['mass_ratio']) ** 0.2 / self.sampled_params['mass_ratio']**0.6
        self.sampled_params['mass_1'] = self.sampled_params['mass_ratio'] * self.sampled_params['mass_2']

        # if self.sampled_params['mass_2'] > self.sampled_params['mass_1']:
        #     self.sampled_params['mass_1'], self.sampled_params['mass_2'] = self.sampled_params['mass_2'], self.sampled_params['mass_1']
        #     self.sampled_params['mass_ratio'] = 1 / self.sampled_params['mass_ratio']


        # # correct units
        # self.sampled_params['mass_2'] *= lal.MSUN_SI
        # self.sampled_params['mass_1'] *= lal.MSUN_SI

        # convert from Bilby convention to Lalsimulation
        self.sampled_params['incl'], self.sampled_params['s1x'], self.sampled_params['s1y'], \
        self.sampled_params['s1z'], self.sampled_params['s2x'], self.sampled_params['s2y'], \
        self.sampled_params['s2z'] = transform_precessing_spins(
            self.sampled_params['theta_jn'], self.sampled_params['phi_jl'],
            self.sampled_params['tilt_1'],
            self.sampled_params['tilt_2'], self.sampled_params['phi_12'],
            self.sampled_params['a_1'], self.sampled_params['a_2'],
            self.sampled_params['mass_1'], self.sampled_params['mass_2'],
            self.sampled_params['reference_frequency'], self.sampled_params['phase']
            )

        self.sampled_params['incl'] = torch.Tensor(self.sampled_params['incl'])
        self.sampled_params['s1x'] = Constant(0).sample((batch_size,)) # torch.Tensor(self.sampled_params['s1x'])
        self.sampled_params['s1y'] = Constant(0).sample((batch_size,)) # torch.Tensor(self.sampled_params['s1y'])
        self.sampled_params['s1z'] = torch.Tensor(self.sampled_params['s1z'])
        self.sampled_params['s2x'] = Constant(0).sample((batch_size,)) # torch.Tensor(self.sampled_params['s2x'])
        self.sampled_params['s2y'] = Constant(0).sample((batch_size,)) # torch.Tensor(self.sampled_params['s2y'])
        self.sampled_params['s2z'] = torch.Tensor(self.sampled_params['s2z'])

        self.sampled_params['f_ref'] = self.sampled_params['reference_frequency']
        self.sampled_params['phiRef'] = self.sampled_params['phase']

        self.sampled_params['dist_mpc'] = (self.sampled_params['dist_mpc'] * u.Mpc).to("m").value # ???

        logger = logging.getLogger(__name__)

        for k in self.sampled_params.keys():
            if type(self.sampled_params[k])==float:
                logger.info(f'The shape of {k} is {self.sampled_params[k]}')
            else:
                logger.info(f'The shape of {k} is {self.sampled_params[k].shape}')
        
        # breakpoint()
        return self.sampled_params