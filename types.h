// Copyright ETH Zurich, IWF
// SPH-FEM cutting simulations on the basis of mfree_iwf-ul_cut_gpu: 
// https://github.com/iwf-inspire/mfree_iwf-ul_cut_gpu
// Date: 2024.03.10

// You should have received a copy of the GNU General Public License
// along with mfree_iwf-ul_cut_gpu.  If not, see <http://www.gnu.org/licenses/>.

// set central solver controls
//	single/double precision
//	method for thermal solution (PSE or Brookshaw, see Eldgredge 2002 and Brookshaw 1994)

#ifndef TYPES_H_
#define TYPES_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <glm/glm.hpp>

#define USE_DOUBLE

#define USE_FEM_TOOL
#define WEAR_NODE_SHIFT
//#define USE_ABRASIVE_WEAR
//#define USE_ABRASIVE_NEW
//#define USE_SUN_WEAR    // Sun et al. user calibrated model. equation (6) and (11). model cannot be implemented... (equation 11)
//#define USE_WHOLE_WEAR
//#define USE_USUI_WEAR    // Usui wear model
//#define USE_USUI_NEW  // updated Usui model
//#define USE_MALA_WEAR    // doesn't work, volume based
//#define USE_ZANGER_WEAR    // combined wear model from Zanger and Schulze
//#define USE_DIFFUSIVE_WEAR    // Diffusive wear model, w_dot = C * exp(-D/T)
//#define USE_RECH_WEAR
#define USE_USUI_RAKE_FLANK

//#define SPARSE_MESH

//#define SURFACE  // for surface convection


#define RESTRICT_WEAR_TEMP

//#define SHEAR_LIMIT_FRI        // for the friction modeling
//#define VELOCITY_FRI
//#define TEMP_FRI
//#define PRES_FRI

#define USE_HOT_TOOL
#define REMESH_GMSH    // for hot tool simulation with segment print, this should be off

#define SMOOTH_TEMP_WEAR

//#define USE_MATERIAL_STRESS

#ifdef USE_DOUBLE

	// built in vector types
	#define float_t double
	#define float2_t double2
	#define float3_t double3
	#define float4_t double4

	// texture types and texture fetching
	#define float_tex_t  int2
	#define float2_tex_t int4
	#define float4_tex_t int4

	#define make_float2_t make_double2
	#define make_float3_t make_double3
	#define make_float4_t make_double4

	#define texfetch1 fetch_double
	#define texfetch2 fetch_double
	#define texfetch4 fetch_double2

	// glm types
    #define mat2x2_t glm::dmat2x2
    #define mat3x3_t glm::dmat3x3
    #define mat3x2_t glm::dmat3x2
    #define mat2x3_t glm::dmat2x3
    #define vec3_t glm::dvec3
    #define vec2_t glm::dvec2

	#define BLOCK_SIZE 256

#else

	// built in vector types
	#define float_t float
	#define float2_t float2
	#define float3_t float3
	#define float4_t float4

	#define make_float2_t make_float2
	#define make_float3_t make_float3
	#define make_float4_t make_float4

	// texture types
	#define float_tex_t  float
	#define float2_tex_t float2
	#define float4_tex_t float4

	#define texfetch1 tex1Dfetch
	#define texfetch2 tex1Dfetch
	#define texfetch4 tex1Dfetch

	// glm types
    #define mat2x2_t glm::mat2x2
    #define mat3x3_t glm::mat3x3
    #define mat3x2_t glm::mat3x2
    #define mat2x3_t glm::mat2x3
    #define vec3_t   glm::vec3
    #define vec2_t   glm::vec2

	#define BLOCK_SIZE 256

#endif

//chose thermal solver
#define Thermal_Conduction_Brookshaw
//#define Thermal_Conduction_PSE

bool check_cuda_error();
bool check_cuda_error(const char *marker);

#endif /* TYPES_H_ */
