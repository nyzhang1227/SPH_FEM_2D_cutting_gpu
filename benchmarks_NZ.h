// Copyright ETH Zurich, IWF
// SPH-FEM cutting simulations on the basis of mfree_iwf-ul_cut_gpu: 
// https://github.com/iwf-inspire/mfree_iwf-ul_cut_gpu
// Date: 2024.03.10

// You should have received a copy of the GNU General Public License
// along with mfree_iwf-ul_cut_gpu.  If not, see <http://www.gnu.org/licenses/>.

// this module contains the SPH-FEM cutting simulations
#ifndef BENCH_NZ
#define BENCH_NZ

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif 
#include <cmath>
#include <thrust/device_vector.h>
#include "constants_structs.h"
#include "particle_gpu.h"
#include "types.h"
#include "actions_gpu.h"
#include "interactions_gpu.h"
#include "tool_FEM.h"
#include "tool_gpu.h"
#include "types.h"
#include "debug.h"
#include "grid.h"
#include "tool_wear.h"
#include "grid_gpu_rothlin.h"
#include "vtk_writer.h"

#include <iostream>
#include <fstream>

extern tool* global_tool;
extern tool_wear* global_wear;
extern tool_FEM* global_tool_FEM;

extern float_t global_dt;
extern float_t global_t_final;

particle_gpu* setup_ref_cut_FEM_tool(int ny, grid_base** grid, float_t rake = 0., float_t clear = 0., float_t chamfer = 0., float_t speed = 0., float_t feed = 0.01);
particle_gpu* setup_ref_cut_FEM_tool_no_wear(int ny, grid_base** grid, float_t rake = 0., float_t clear = 0., float_t chamfer = 0., float_t speed = 0., float_t feed = 0.01);
particle_gpu* setup_ref_cut_FEM_tool_gmsh_friction_paper_try(int ny, grid_base** grid, float_t rake = 0., float_t clear = 0., float_t chamfer = 0., float_t speed = 0., float_t feed = 0.01);
particle_gpu* setup_ref_cut_FEM_tool_gmsh_Ti64(int ny, grid_base** grid, float_t rake = 0., float_t clear = 0., float_t chamfer = 0., float_t speed = 0., float_t feed = 0.01);
particle_gpu* setup_ref_cut_FEM_tool_gmsh_Ck45(int ny, grid_base** grid, float_t rake = 0., float_t clear = 0., float_t chamfer = 0., float_t speed = 0., float_t feed = 0.01);
particle_gpu* setup_ref_cut_FEM_tool_gmsh_friction_paper_try_large_feed(int ny, grid_base** grid, float_t rake = 0., float_t clear = 0., float_t chamfer = 0., float_t speed = 0., float_t feed = 0.01);
particle_gpu* setup_ref_cut_FEM_tool_gmsh_wear_test_large_feed_Ti64(int ny, grid_base** grid, float_t rake = 0., float_t clear = 0., float_t chamfer = 0., float_t speed = 0., float_t feed = 0.01);
particle_gpu* setup_ref_cut_FEM_tool_gmsh_wear_test(int ny, grid_base** grid, float_t rake = 0., float_t clear = 0., float_t chamfer = 0., float_t speed = 0., float_t feed = 0.01);
particle_gpu* setup_ref_cut_FEM_tool_gmsh_wear_test_Ck45(int ny, grid_base** grid, float_t rake = 0., float_t clear = 0., float_t chamfer = 0., float_t speed = 0., float_t feed = 0.01);
particle_gpu* setup_ref_cut_FEM_tool_gmsh_wear_test_Ck45_flank(int ny, grid_base** grid, float_t rake = 0., float_t clear = 0., float_t chamfer = 0., float_t speed = 0., float_t feed = 0.01);
particle_gpu* setup_ref_cut_FEM_tool_no_wear_thermal_paper(int ny, grid_base** grid, float_t rake = 0., float_t clear = 0., float_t chamfer = 0., float_t speed = 0., float_t feed = 0.01);
particle_gpu* setup_ref_cut_FEM_tool_textured(int ny, grid_base** grid, float_t rake = 0., float_t clear = 0., float_t chamfer = 0., float_t speed = 0., float_t feed = 0.01);

#endif