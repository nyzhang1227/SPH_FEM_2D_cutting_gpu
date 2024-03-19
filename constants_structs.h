// Copyright ETH Zurich, IWF
// SPH-FEM cutting simulations on the basis of mfree_iwf-ul_cut_gpu: 
// https://github.com/iwf-inspire/mfree_iwf-ul_cut_gpu
// Date: 2024.03.10

// You should have received a copy of the GNU General Public License
// along with mfree_iwf-ul_cut_gpu.  If not, see <http://www.gnu.org/licenses/>.

// this module defines collections of constants (e.g. physical constants, correction constants etc.)

#ifndef CONSTANTS_STRUCTS_H_
#define CONSTANTS_STRUCTS_H_

#include "types.h"

#include <cstring>

struct phys_constants {
	float_t E;
	float_t nu;
	float_t rho0;
	float_t K;
	float_t G;
	float_t mass;
	float_t mass_tool;
};

phys_constants make_phys_constants();

struct trml_constants {
	float_t cp; //thermal capacity
	float_t tq;	//taylor guinnie
	float_t k;	//thermal conductivity
	float_t alpha;	//thermal diffusitivty
	float_t T_init;	//initial temperature
	float_t eta;	//fraction of frictional power turned to heat
};

trml_constants make_trml_constants();

struct corr_constants {
	float_t wdeltap;
	float_t stresseps;
	float_t xspheps;
	float_t alpha;
	float_t beta;
	float_t eta;
};

corr_constants make_corr_constants();

struct joco_constants {
	float_t A;
	float_t B;
	float_t C;
	float_t n;
	float_t m;
	float_t Tmelt;
	float_t Tref;
	float_t eps_dot_ref;

	float_t tanh_a;
	float_t tanh_b;
	float_t tanh_c;
	float_t tanh_d;
};

joco_constants make_joco_constants();

struct geom_constants {
	int nx;
	int ny;
	float_t bbmin_x;
	float_t bbmin_y;
	float_t dx;
};

geom_constants make_geom_constants();

struct tool_constants {
	float_t cp; //thermal capacity
	float_t tq;	//taylor guinnie
	float_t k;	//thermal conductivity
	float_t eta;	//fraction of frictional power turned to heat
	float_t rho;    // density
	float_t T0;     // Fixed boundary temperature
	float_t h;      // heat transfer coefficient (convection) between tool and wp
	float_t beta;   // proportion of generated frictional heat into the workpiece material
	float_t mu;     // frictional coefficient
	float_t contact_alpha; // contact stiffness
	float_t slave_mass;    // wp particle mass
	float_t hc;  // heat convection coefficient
#ifdef SHEAR_LIMIT_FRI
	float_t shear_limit;
#endif // SHEAR_LIMIT_FRI
#ifdef VELOCITY_FRI
	float_t exp_co_vel;
#endif // VELOCITY_FRI
#ifdef TEMP_FRI
	float_t exp_co_temp;
	float_t T_melt;
#endif // TEMP_FRI
#ifdef PRES_FRI
    float_t exp_co_pres;
	float_t up_limit;
	float_t lo_limit;
#endif

};

tool_constants make_tool_constants();

struct wear_constants {
	float_t A; 
	float_t B;
	float_t C;
	float_t D;

};

wear_constants make_wear_constants();

#endif /* CONSTANTS_STRUCTS_H_ */
