// Copyright ETH Zurich, IWF
// SPH-FEM cutting simulations on the basis of mfree_iwf-ul_cut_gpu: 
// https://github.com/iwf-inspire/mfree_iwf-ul_cut_gpu
// Date: 2024.03.10

// You should have received a copy of the GNU General Public License
// along with mfree_iwf-ul_cut_gpu.  If not, see <http://www.gnu.org/licenses/>.

// module for mesh structure

#ifndef MS2D_GPU
#define MS2D_GPU

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <iterator>
#include <algorithm>
#include <set>
#include <gmsh.h>

#include "types.h"
#include "glm/glm.hpp"




#define FEM_BLOCK_SIZE 256
#define FEM_ASSEMB_SIZE 16

struct elements_read {

        std::vector<int4> node_num;  // x,y,y for no. of three nodes, w for the color
        //std::vector<int> color_map;
        std::vector<int3> cutting_boundary_flags;
        std::vector<int3> fixed_boundary_flags;
        std::vector<int> cutting_boundary_elements;
        std::vector<int> fixed_boundary_elements;
        std::vector<bool> cutting_boundary_contact;             // Size of this vector is equal to the size of cutting_boundary_elements
        std::vector<float3_t> contact_element_temperature;        // Same size as above
        std::vector<float3_t> contact_particle_number;               // Same size as above
        std::vector<float3_t> frictional_heat;
        std::vector<int2> boundary_layers;
        //elements_read();
};

struct nodes_read {
    std::vector<float_t> x_init;
    std::vector<float_t> y_init;   
    std::vector<int> cutting_edge_flag;
    std::vector<int> fixed_edge_flag;
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
#ifdef USE_HOT_TOOL
    std::vector<float_t> T_init;
#endif
    //nodes_read();
};

class mesh_read {
public:
    nodes_read nodes;
    elements_read elements;
    int color_total =18;        // total number of colors, 12 for the tool
    int color_cutting_edge = 6;
    mesh_read(std::string file_name, bool gmsh_d);
    mesh_read(std::string file_name);
    mesh_read();

    void mesh_read_temperature(std::string file_name);
    void write_mesh_file();
};

class mesh_GPU_m{
private:
    int ele_num_total = 0;
    int ele_num_cutting = 0;
    int ele_num_fixed = 0;
    int no_num = 0;
    int color_total = 0;
    int cutting_edge_color = 0;
    int non_zeros = 0;

public:
    // element related
	int4 *node_num = 0;
    //int *color_map = 0;
	int3 *cutting_boundary_flags = 0;
	int3 *fixed_boundary_flags = 0;
    int *cutting_boundary_elements = 0;
    int *fixed_boundary_elements = 0;
#ifdef WEAR_NODE_SHIFT
    int2* cutting_boundary_layer = 0;
#endif
    //int *cutting_boundary_contact = 0;        // boolean type
    //float3_t *contact_element_temperature = 0;        // Same size as above
    //float3_t *contact_particle_number = 0; 
    //float3_t *contact_frictional_heat = 0; 
    //int *contact_particle_No_edge_x = 0;
    //int *contact_particle_No_edge_y = 0;
    //int *contact_particle_No_edge_z = 0;

    // nodes related
    float2_t *position = 0;          // position
    int *cutting_edge_flag = 0;    // flag of cutting edge
    int *fixed_edge_flag = 0;      // flag of fixed edge
    float_t *T_nodes = 0;                // nodes temperature
    //float_t *T_nodes_old = 0;            // nodes temperature at previous step
    float_t *f = 0;                      // source item of nodes
    float_t *k = 0;                      // item of discrete evolution operator for each nodes
#ifdef WEAR_NODE_SHIFT
    float2_t* position_mapping = 0;
    int4* node_num_mapping = 0;
#endif

    // CSR matrix related, with zero-based indexing
    int *csr_row_ptr = 0;
    int *csr_col_ind = 0;

    //mesh_GPU(nodes_read& mesh, elements_read& elements);
    mesh_GPU_m();
    mesh_GPU_m(nodes_read nodes, elements_read elements, int color_number, int cut_edge_col_no, bool ini);

    int get_node_number();
    int get_element_number();
    int get_cutting_edge_element_number();
    int get_fixed_edge_element_number();
    int get_non_zero_number();
    int get_color_number();
    int get_cutting_edge_color_number();
    void reset_mesh_gpu(mesh_GPU_m* mesh_GPU_new);

    //void vector_adjustment(int a);
    //void T_vector_adjustment(float_t a);

};

int number_to_jump(int a, std::vector<int>exclude_node);
void generate_tool_mesh_gmsh(std::vector<vec2_t> points, float_t r, float_t lc);
// namespace functions
//mat3x2_t get_element_coordinates_2D(float2_t a, float2_t b, float2_t c);   // 3 columns and 2 rows
//mat3x2_t get_element_coordinates_2D(int3 nom, float2_t * nodes);   // 3 columns and 2 rows

//float_t get_element_area_2D(mat3x2_t coordinates);

//float3_t get_element_edge_length_2D(mat3x2_t coordinates);

#endif // MS2D_GPU