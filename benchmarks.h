// Copyright ETH Zurich, IWF
// SPH-FEM cutting simulations on the basis of mfree_iwf-ul_cut_gpu: 
// https://github.com/iwf-inspire/mfree_iwf-ul_cut_gpu
// Date: 2024.03.10

// You should have received a copy of the GNU General Public License
// along with mfree_iwf-ul_cut_gpu.  If not, see <http://www.gnu.org/licenses/>.

//this module contains the complete code to setup metal cutting simulations and some preliminary benchmarks

#ifndef BENCHMARKS_H_
#define BENCHMARKS_H_

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif 


#include <thrust/device_vector.h>
#include "constants_structs.h"
#include "particle_gpu.h"
#include "types.h"
#include "actions_gpu.h"
#include "interactions_gpu.h"
#include "tool.h"
#include "tool_gpu.h"
#include "types.h"
#include "debug.h"
#include "grid.h"
#include "tool_wear.h"
#include "grid_gpu_rothlin.h"

#include <iostream>
#include <fstream>

extern tool *global_tool;
extern tool_wear *global_wear;
extern tool_FEM* global_tool_FEM;

//rubber ring impact, see for example gray & monaghan 2001 (verifies elastic stage)
//		this example uses SI units
particle_gpu *setup_rings(int nbox, grid_base **grid);
//plastic - plastic wall impact. see for example rothlin 2019 (verifies plastic stage)
//		this example uses SI units
particle_gpu *setup_impact(int nbox, grid_base **grid);
//reference cut as defined by ruttimann 2012
//		this example uses bomb units!!!! [musec, kg, cm]
particle_gpu *setup_ref_cut(int ny, grid_base **grid, float_t rake = 0., float_t clear = 0., float_t chamfer = 0., float_t speed = 0., float_t feed = 0.01);

#endif /* BENCHMARKS_H_ */
