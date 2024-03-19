// Copyright ETH Zurich, IWF
// SPH-FEM cutting simulations on the basis of mfree_iwf-ul_cut_gpu: 
// https://github.com/iwf-inspire/mfree_iwf-ul_cut_gpu
// Date: 2024.03.10

// You should have received a copy of the GNU General Public License
// along with mfree_iwf-ul_cut_gpu.  If not, see <http://www.gnu.org/licenses/>.

// leap frog timestepper

#ifndef LEAP_FROG_H_
#define LEAP_FROG_H_

#include "grid.h"
#include "particle_gpu.h"

#include "actions_gpu.h"
#include "interactions_gpu.h"
#include "tool_gpu.h"
#include "tool_FEM.h"

#include "types.h"

#include "device_launch_parameters.h"

extern tool *global_tool;
extern tool_FEM* global_tool_FEM;
extern int global_step;
extern float_t global_dt;

class leap_frog{
private:
	float2_t *pos_init = 0;
	float2_t *vel_init = 0;
	float4_t *S_init   = 0;
	float_t  *rho_init = 0;
	float_t  *T_init   = 0;

	int *cell_start = 0;
	int *cell_end   = 0;

	float_t* T_tool_init;


public:
	void step(particle_gpu *particles, grid_base *g);
	void step(particle_gpu* particles, grid_base* g, tool_FEM* m_tool);
	leap_frog(unsigned int num_part, unsigned int num_cell);
	leap_frog(unsigned int num_part, unsigned int num_cell, int node_num_in_FEM_tool);
};

#endif /* LEAP_FROG_H_ */
