from nipype.interfaces.ants import N4BiasFieldCorrection
import sys
import ast

if len(sys.argv) < 2:
    print("INPUT from ipython: run n4_bias_correction "
          "input_image dimension n_iterations(optional,"
          " form:[n_1,n_2,n_3,n_4]) output_image(optional)")
    sys.exit(1)

# if output_image is given
if len(sys.argv) > 3:
    n4 = N4BiasField