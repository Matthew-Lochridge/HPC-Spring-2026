import numpy as np

# Configuration parameters
class parameters:

    def __init__(self):

        # Physical constants
        self.Ry = 13.6 # Rydberg energy (eV)
        self.r_H = 5.2917725e-11 # Bohr radius (m)
        self.r_C = 73e-12 / self.r_H # atomic radius of sp2-bonded carbon (Bohr radii)
        self.a_CC = 1.42e-10 / self.r_H # carbon-carbon bond length in graphene (Bohr radii)
        self.a_CH = 1.0919e-10 / self.r_H # carbon-hydrogen bond length in methane (Bohr radii)
        self.a_vdW = 3.35e-10 / self.r_H # interlayer spacing in graphite (Bohr radii)
        self.a_d = 3.56683e-10 / self.r_H # lattice constant of diamond (Bohr radii)

        # Kurokawa pseudopotential parameters for carbon
        self.b_C = np.array([1.781, 1.424, 0.354, 0.938])

        # Kurokawa pseudopotential parameters for hydrogen
        self.b_H = np.array([-0.397, 0.0275, 0.1745, -0.0531, 0.0811, -1.086, 2.71, -2.86])

        # Nanostructure configuration parameters
        self.N_a = 1 # number of primitive helical motifs along a tube within the supercell (axial) Set to 1 for an infinite tube.
        self.N_x = 0 # number of primitive graphene translations between ribbons or tubes (axial) Set to 0 for an infinite ribbon or tube.
        self.N_y = 4*np.sqrt(3)*self.a_CC # number of primitive graphene translations between ribbons (transverse in-plane)
        self.N_z = 3*np.sqrt(3)*self.a_CC # number of primitive graphene translations between graphene sheets or ribbons (transverse out-of-plane) or tubes (side length of square cross-section)
        
        # Other parameters
        self.E_cut = 15 # cutoff energy (Ry)
        self.max_G = 100 # maximum Manhattan distance of G vectors
        self.n_x = 101 # number of real-space points
        self.n_k = 101 # number of k-space points
