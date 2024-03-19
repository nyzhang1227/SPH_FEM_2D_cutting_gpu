// Copyright ETH Zurich, IWF
// SPH-FEM cutting simulations on the basis of mfree_iwf-ul_cut_gpu: 
// https://github.com/iwf-inspire/mfree_iwf-ul_cut_gpu
// Date: 2024.03.10

// You should have received a copy of the GNU General Public License
// along with mfree_iwf-ul_cut_gpu.  If not, see <http://www.gnu.org/licenses/>.

#include "mesh_structure_GPU.h"

extern bool wear_pro;

int number_to_jump(int a, std::vector<int>exclude_node){
    int jump = 0;
	for (int i = 0; i < exclude_node.size(); i++){
       if (exclude_node[i] < a) jump++;
	}
    return jump;
}

mesh_GPU_m::mesh_GPU_m()
{
}

mesh_GPU_m::mesh_GPU_m(nodes_read nodes, elements_read  elements, int color_number, int cutting_edge_color_no, bool ini){
	
    // nodes related variables
    // Create buffer for data transfer
    int N = nodes.x_init.size();  // number of nodes
	this->no_num = N;
    float2_t *pos_h = new float2_t[N];
    int *cutting_edge_flag_h = new int[N];    
    int *fixed_edge_flag_h = new int[N];     
    float_t *T_nodes_h = new float_t[N];
    float_t *T_nodes_old_h = new float_t[N];    


    for (int i = 0; i < N; i++){
        pos_h[i].x = nodes.x_init[i];
        pos_h[i].y = nodes.y_init[i];
        cutting_edge_flag_h[i] = nodes.cutting_edge_flag[i];
        fixed_edge_flag_h[i] = nodes.fixed_edge_flag[i];
        T_nodes_h[i] = 300.;
        T_nodes_old_h[i] = 300.;

#ifdef USE_HOT_TOOL
        if (ini){
		    T_nodes_h[i] = nodes.T_init[i];
		    T_nodes_old_h[i] = nodes.T_init[i];
		}
#endif
	}

	cudaMalloc((void **) &position, sizeof(float2_t)*N);
    cudaMalloc((void **) &cutting_edge_flag, sizeof(int)*N);
    cudaMalloc((void **) &fixed_edge_flag, sizeof(int)*N);
    cudaMalloc((void **) &T_nodes, sizeof(float_t)*N);
    cudaMalloc((void **) &f, sizeof(float_t)*N);
    cudaMalloc((void **) &k, sizeof(float_t)*N);
#ifdef WEAR_NODE_SHIFT
	cudaMalloc((void**) &position_mapping, sizeof(float2_t) * N);
#endif

    cudaMemcpy(position, pos_h, sizeof(float2_t) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cutting_edge_flag, cutting_edge_flag_h, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(fixed_edge_flag, fixed_edge_flag_h, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(T_nodes, T_nodes_h, sizeof(float_t) * N, cudaMemcpyHostToDevice);

    cudaMemset(f, 0, sizeof(float_t) * N);
    cudaMemset(k, 0, sizeof(float_t) * N);

#ifdef WEAR_NODE_SHIFT
	cudaMemcpy(position_mapping, pos_h, sizeof(float2_t) * N, cudaMemcpyHostToDevice);
#endif


    delete [] pos_h;
    delete [] cutting_edge_flag_h;
    delete [] fixed_edge_flag_h;
    delete [] T_nodes_h;

    // elements related variables
    // Create buffer for data transfer

	color_total = color_number;
	cutting_edge_color = cutting_edge_color_no;

    int e_total = elements.node_num.size();  // number of nodes
    int e_cutting = elements.cutting_boundary_elements.size();
    int e_fixed = elements.fixed_boundary_elements.size();

    this->ele_num_total = e_total;
	this->ele_num_cutting = e_cutting;
	this->ele_num_fixed = e_fixed;

    int4 *node_num_h = new int4[e_total];
	int3 *cutting_boundary_flags_h = new int3[e_total];
	int3 *fixed_boundary_flags_h = new int3[e_total];
    int *cutting_boundary_elements_h = new int[e_cutting];
    int *fixed_boundary_elements_h = new int[e_fixed];
#ifdef WEAR_NODE_SHIFT
	int2* cutting_boundary_layer_h = new int2[e_total];
#endif

    for (int i = 0; i < e_total; i++){
        node_num_h[i] = elements.node_num[i];

        cutting_boundary_flags_h[i] = elements.cutting_boundary_flags[i];
        fixed_boundary_flags_h[i] = elements.fixed_boundary_flags[i];
#ifdef WEAR_NODE_SHIFT
		cutting_boundary_layer_h[i] = elements.boundary_layers[i];
#endif
	}

    for (int j = 0; j < e_cutting; j++){
        cutting_boundary_elements_h[j] = elements.cutting_boundary_elements[j];
    }

    for (int k = 0; k < e_fixed; k++){
        fixed_boundary_elements_h[k] = elements.fixed_boundary_elements[k];
    }

    cudaMalloc((void **) &node_num, sizeof(int4) * e_total);
    cudaMalloc((void **) &cutting_boundary_flags, sizeof(int3) * e_total);
    cudaMalloc((void **) &fixed_boundary_flags, sizeof(int3) * e_total);
#ifdef WEAR_NODE_SHIFT
	cudaMalloc((void**) &cutting_boundary_layer, sizeof(int2) * e_total);
	cudaMalloc((void**) &node_num_mapping, sizeof(int4) * e_total);
#endif
    cudaMalloc((void **) &cutting_boundary_elements, sizeof(int) * e_cutting);
    cudaMalloc((void **) &fixed_boundary_elements, sizeof(int) * e_fixed);


    cudaMemcpy(node_num, node_num_h, sizeof(int4) * e_total, cudaMemcpyHostToDevice);
    cudaMemcpy(cutting_boundary_flags, cutting_boundary_flags_h, sizeof(int3) * e_total, cudaMemcpyHostToDevice);
    cudaMemcpy(fixed_boundary_flags, fixed_boundary_flags_h, sizeof(int3) * e_total, cudaMemcpyHostToDevice);
    cudaMemcpy(cutting_boundary_elements, cutting_boundary_elements_h, sizeof(int) * e_cutting, cudaMemcpyHostToDevice);
    cudaMemcpy(fixed_boundary_elements, fixed_boundary_elements_h, sizeof(int) * e_fixed, cudaMemcpyHostToDevice);
#ifdef WEAR_NODE_SHIFT
	cudaMemcpy(cutting_boundary_layer, cutting_boundary_layer_h, sizeof(int2) * e_total, cudaMemcpyHostToDevice);
	cudaMemcpy(node_num_mapping, node_num_h, sizeof(int4) * e_total, cudaMemcpyHostToDevice);
#endif

    delete [] node_num_h;
    delete [] cutting_boundary_flags_h;
    delete [] fixed_boundary_flags_h;
    delete [] cutting_boundary_elements_h;
    delete [] fixed_boundary_elements_h;
#ifdef WEAR_NODE_SHIFT
	delete[] cutting_boundary_layer_h;
#endif
	
	// CSR matrix related variables
	non_zeros = nodes.col_ind.size();

	int* csr_row_ptr_h = new int[no_num + 1]; 
	int* csr_col_ind_h = new int[non_zeros];


	for (int k = 0; k < no_num + 1; k++) {
		csr_row_ptr_h[k] = nodes.row_ptr[k];
	}

	for (int kk = 0; kk < non_zeros; kk++) {
		csr_col_ind_h[kk] = nodes.col_ind[kk];	
	}

	cudaMalloc((void **) &csr_row_ptr, sizeof(int) * (no_num + 1));
	cudaMalloc((void **) &csr_col_ind, sizeof(int) * non_zeros);

	cudaMemcpy(csr_row_ptr, csr_row_ptr_h, sizeof(int) * (no_num + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(csr_col_ind, csr_col_ind_h, sizeof(int) * non_zeros, cudaMemcpyHostToDevice);

	delete[] csr_row_ptr_h;
	delete[] csr_col_ind_h;
}

int mesh_GPU_m::get_node_number()
{
	return no_num;
}

int mesh_GPU_m::get_element_number()
{
	return ele_num_total;
}

int mesh_GPU_m::get_cutting_edge_element_number()
{
	return ele_num_cutting;
}

int mesh_GPU_m::get_fixed_edge_element_number()
{
	return ele_num_fixed;
}

int mesh_GPU_m::get_non_zero_number()
{
	return non_zeros;
}

int mesh_GPU_m::get_color_number()
{
	return color_total;
}

int mesh_GPU_m::get_cutting_edge_color_number()
{
	return cutting_edge_color;
}

mesh_read::mesh_read(std::string file_name, bool gmsh_d)
{
	std::ifstream infile(file_name, std::ifstream::in);
    //printf("Read mesh starts\n");
	std::string line;
	std::string NodeSet("Line");    // indicator for node sets of boundaries
	std::string ElementSet("ELSET");  // element sets for boundaries
	std::string NodeID("*NODE");  // Start of node
	std::string ElementID("type=CPS3"); // start of elements
	std::string CuttingEdgeID("CuttingEdge");
	std::string FixedEdgeID("FixedBoundary");
	std::string StopID("End");
	std::string EdgeElementID("T3D2");
	std::string comment_id("#");
	std::string node_finish_id("*******");

	bool NodeStart = 0;
	bool ElementStart = 0;
	bool EdgeElementStart = 0;
	bool CuttingEdgeInElements = 0;
	bool FixedEdgeInElements = 0;
	int NodeNumber = 0;
	int ElementNumber = 0;


	std::vector<float_t> seq_;
	std::vector<float_t> x_;
	std::vector<float_t> y_;
	std::vector<float_t> ele_seq_;
	std::vector<int> CuttingEdgeNum;
	std::vector<int2> CuttingEdgeWithNode;
	std::vector<int> FixedEdgeNum;
	std::vector<int2> FixedEdgeWithNode;
	std::vector<int2> boundary_edge;
	std::vector<int> CuttingBoundaryCorrection;
	std::vector<int> FixedBoundaryCorrection;
	std::vector<int> buffer_node_in_elements;

	std::vector<int> exclude_node;

	while (std::getline(infile, line)){
		
		if (line.find(ElementID) != std::string::npos) {
			ElementStart = 1;
			NodeStart = 0;
			EdgeElementStart = 0;
			CuttingEdgeInElements = 0;
			FixedEdgeInElements = 0;
			continue;
		}

		if (line.find(FixedEdgeID) != std::string::npos ) {
			CuttingEdgeInElements = 0;
			FixedEdgeInElements = 1;
			EdgeElementStart = 0;
			NodeStart = 0;
			ElementStart = 0;
			break;
		}

		if (ElementStart) {

			std::istringstream ss(line);
			std::string token;
			std::vector<std::string> buf;
			int a;
			while (std::getline(ss, token, ',')) {
				buf.push_back(token);
			}
			for (int i = 1; i < 4; i++) {
				std::stringstream nu(buf[i]);
				nu >> a;
				buffer_node_in_elements.push_back(a - 1);
			}
		}

	}
    std::set<int> s_nodes;
	unsigned size_node_elements = buffer_node_in_elements.size();
	for(unsigned i = 0; i < size_node_elements; i++) s_nodes.insert(buffer_node_in_elements[i]);
	buffer_node_in_elements.assign(s_nodes.begin(), s_nodes.end());

    //printf("Read nodes in elements successful\n");
	CuttingEdgeInElements = 0;
	FixedEdgeInElements = 0;
	EdgeElementStart = 0;
	NodeStart = 0;
	ElementStart = 0;
    std::ifstream infile1(file_name, std::ifstream::in);
	while (std::getline(infile1, line)) {

		if (line.find(comment_id) != std::string::npos ) {
			printf("Found a comment line \n");
			// npos is a static member constant value with the greatest possible value for an element of type size_t. contains "find" at least once.
			continue;
		}

		if (line.find(NodeID) != std::string::npos) {
			NodeStart = 1;
			continue;
		}

		if (line.find(EdgeElementID) != std::string::npos) {
			EdgeElementStart = 1;
			CuttingEdgeInElements = 0;
			FixedEdgeInElements = 0;
			NodeStart = 0;
			ElementStart = 0;
			continue;
		}
		
		if (line.find(ElementID) != std::string::npos){
			EdgeElementStart = 0;
			CuttingEdgeInElements = 0;
			FixedEdgeInElements = 0;
			NodeStart = 0;
			ElementStart = 0;
			continue;
		}

		if (line.find(node_finish_id) != std::string::npos) {
			NodeStart = 0;
			EdgeElementStart = 0;
			CuttingEdgeInElements = 0;
			FixedEdgeInElements = 0;
			ElementStart = 0;
            //printf("Read node finished\n");
			int n = x_.size();
			int zero = 0;
			//printf("nn is %d, NodeNumber is %d \n", n, NodeNumber);
			assert(n == NodeNumber);
			nodes.x_init.resize(n);
            nodes.y_init.resize(n);
			nodes.cutting_edge_flag.resize(n);
			nodes.fixed_edge_flag.resize(n);
			for (int i = 0; i < n; i++) {
				nodes.x_init[seq_[i]] = x_[i] ;
				nodes.y_init[seq_[i]] = y_[i] ;
				nodes.cutting_edge_flag[seq_[i]] = zero;
				nodes.fixed_edge_flag[seq_[i]] = zero;
				//printf("node number is %u, x %f, y %f, cutting_edge %u, fixed_edge %u \n", i, nodes.x_init[i], nodes.y_init[i], nodes.cutting_edge_flag[i], nodes.fixed_edge_flag[i]);
			}
			continue;
		}

		if (line.find(CuttingEdgeID) != std::string::npos ) {
			CuttingEdgeInElements = 1;
			FixedEdgeInElements = 0;
			EdgeElementStart = 0;
			NodeStart = 0;
            ElementStart = 0;
			continue;
		}

		if (line.find(FixedEdgeID) != std::string::npos ) {
			CuttingEdgeInElements = 0;
			FixedEdgeInElements = 1;
			EdgeElementStart = 0;
			NodeStart = 0;
			ElementStart = 0;
			continue;
		}

		if (NodeStart) {			
			std::istringstream ss(line);
			int a;
			float_t x;
			float_t y;
			std::size_t offset = 0;
			std::string token;
			std::getline(ss, token, ',');
			a = std::stoi(token, &offset);
			
			std::vector<int>::iterator it0;
			it0 = std::find(buffer_node_in_elements.begin(), buffer_node_in_elements.end(), a-1);
			if (it0 == buffer_node_in_elements.end()) {
               exclude_node.push_back(a);
			   printf("Node %d is excluded \n", a);
               continue;
			}

			//std::stringstream nu(token);
			//nu >> a;
			int j = number_to_jump(a, exclude_node);
			std::getline(ss, token, ',');
			x = std::stod(token, &offset);
			std::getline(ss, token, ',');
			y = std::stod(token, &offset);
			seq_.push_back(a-1-j);
			x_.push_back(x);
			y_.push_back(y);
			NodeNumber++;
			ss.clear();
			//printf("Node number is %d\n", NodeNumber);
		}	

		if (EdgeElementStart){
			std::istringstream ss(line);
			std::string token;
			std::vector<std::string> buf;
			std::vector<int> bufint;
			int a;
			while (std::getline(ss, token, ',')) {
				buf.push_back(token);
			}
			int2 buff;
			for (int i = 1; i < 3; i++) {
				std::stringstream nu(buf[i]);
				nu >> a;
				int j = number_to_jump(a, exclude_node);
				bufint.push_back(a - 1 - j);
			}
			buff.x = bufint[0];
			buff.y = bufint[1];
			boundary_edge.push_back(buff);
		}

		if (CuttingEdgeInElements) {
			std::istringstream ss(line);
			int a;
			std::string token;
			int2 buffer_node;
			while (std::getline(ss, token, ',')) {
				std::stringstream nu(token);
				nu >> a;
                buffer_node = boundary_edge[a-1];
				int j1 = number_to_jump(buffer_node.x, exclude_node);
				int j2 = number_to_jump(buffer_node.y, exclude_node);
				int2 buffer_edge;
				buffer_edge.x = buffer_node.x  - j1;
				buffer_edge.y = buffer_node.y  - j2;
				nodes.cutting_edge_flag[buffer_edge.x] = 1;
				CuttingEdgeNum.push_back(buffer_edge.x);	
				nodes.cutting_edge_flag[buffer_edge.y] = 1;
				CuttingEdgeNum.push_back(buffer_edge.y);	

				CuttingEdgeWithNode.push_back(buffer_edge);		
			}
			ss.clear();
		}

		if (FixedEdgeInElements) {
			std::istringstream ss(line);
			int a;
			std::string token;
			int2 buffer_node;
			while (std::getline(ss, token, ',')) {
				std::stringstream nu(token);
				nu >> a;
                buffer_node = boundary_edge[a-1];
				int j1 = number_to_jump(buffer_node.x, exclude_node);
				int j2 = number_to_jump(buffer_node.y, exclude_node);
				int2 buffer_edge;
				buffer_edge.x = buffer_node.x  - j1;
				buffer_edge.y = buffer_node.y  - j2;
				nodes.fixed_edge_flag[buffer_edge.x] = 1;
				FixedEdgeNum.push_back(buffer_edge.x);	
				nodes.fixed_edge_flag[buffer_edge.y] = 1;
				FixedEdgeNum.push_back(buffer_edge.y);		

				FixedEdgeWithNode.push_back(buffer_edge);			
			}
			ss.clear();
		}

	}

	ElementStart = 0;
	NodeStart = 0;
    EdgeElementStart = 0;
    CuttingEdgeInElements = 0;
	FixedEdgeInElements = 0;

	// Now eliminate repeating items in boundary edge vectors
	std::set<int> s_c;
	unsigned size_cutting_edge = CuttingEdgeNum.size();
	for(unsigned i = 0; i < size_cutting_edge; i++) s_c.insert(CuttingEdgeNum[i]);
	CuttingEdgeNum.assign(s_c.begin(), s_c.end());



	std::set<int> s_f;
	unsigned size_fixed_edge = FixedEdgeNum.size();
	for(unsigned i = 0; i < size_fixed_edge; i++) s_f.insert(FixedEdgeNum[i]);
	FixedEdgeNum.assign(s_f.begin(), s_f.end());

	std::ifstream infile2(file_name, std::ifstream::in);
	while (std::getline(infile2, line)){	

		if (line.find(ElementID) != std::string::npos) {
			ElementStart = 1;
			NodeStart = 0;
            EdgeElementStart = 0;
			CuttingEdgeInElements = 0;
			FixedEdgeInElements = 0;
			
			continue;
		}

		if (line.find(FixedEdgeID) != std::string::npos ) {
			CuttingEdgeInElements = 0;
			FixedEdgeInElements = 0;
			EdgeElementStart = 0;
			NodeStart = 0;
			ElementStart = 0;
			break;
		}

    	if (ElementStart) {

			std::istringstream ss(line);
			std::string token;
			std::vector<std::string> buf;
			std::vector<int> bufint;
			int a;
			while (std::getline(ss, token, ',')) {
				buf.push_back(token);
			}
			int4 buff;
			for (int i = 1; i < 4; i++) {
				std::stringstream nu(buf[i]);
				nu >> a;
				int j = number_to_jump(a, exclude_node);
				bufint.push_back(a - 1 - j);
			}
			buff.x = bufint[0];
			buff.y = bufint[1];
			buff.z = bufint[2];
			buff.w = 0;
			elements.node_num.push_back(buff);

            int3 BoundaryIni_c; //MeshStructure::tri_boundary BoundaryIni;
	        BoundaryIni_c.x = 0;
	        BoundaryIni_c.y = 0;
	        BoundaryIni_c.z = 0;

			int3 BoundaryIni_f; //MeshStructure::tri_boundary BoundaryIni;
	        BoundaryIni_f.x = 0;
	        BoundaryIni_f.y = 0;
	        BoundaryIni_f.z = 0;


			for (int i = 0; i < CuttingEdgeWithNode.size(); i++){
				if ((CuttingEdgeWithNode[i].x == buff.x && CuttingEdgeWithNode[i].y == buff.y) || (CuttingEdgeWithNode[i].x == buff.y && CuttingEdgeWithNode[i].y == buff.x)) { BoundaryIni_c.x = 1;}
				if ((CuttingEdgeWithNode[i].x == buff.y && CuttingEdgeWithNode[i].y == buff.z) || (CuttingEdgeWithNode[i].x == buff.z && CuttingEdgeWithNode[i].y == buff.y)) { BoundaryIni_c.y = 1;}
				if ((CuttingEdgeWithNode[i].x == buff.z && CuttingEdgeWithNode[i].y == buff.x) || (CuttingEdgeWithNode[i].x == buff.x && CuttingEdgeWithNode[i].y == buff.z)) { BoundaryIni_c.z = 1;}
				
			}

			elements.cutting_boundary_flags.push_back(BoundaryIni_c);
			if (BoundaryIni_c.x == 1 || BoundaryIni_c.y == 1|| BoundaryIni_c.z == 1) {
				elements.cutting_boundary_elements.push_back(ElementNumber);
                elements.cutting_boundary_contact.push_back(0);
			}

			
			for (int i = 0; i < FixedEdgeWithNode.size(); i++){
				if ((FixedEdgeWithNode[i].x == buff.x && FixedEdgeWithNode[i].y == buff.y) || (FixedEdgeWithNode[i].x == buff.y && FixedEdgeWithNode[i].y == buff.x)) { BoundaryIni_f.x = 1;}
				if ((FixedEdgeWithNode[i].x == buff.y && FixedEdgeWithNode[i].y == buff.z) || (FixedEdgeWithNode[i].x == buff.z && FixedEdgeWithNode[i].y == buff.y)) { BoundaryIni_f.y = 1;}
				if ((FixedEdgeWithNode[i].x == buff.z && FixedEdgeWithNode[i].y == buff.x) || (FixedEdgeWithNode[i].x == buff.x && FixedEdgeWithNode[i].y == buff.z)) { BoundaryIni_f.z = 1;}
				
			}


			elements.fixed_boundary_flags.push_back(BoundaryIni_f);

			if (BoundaryIni_f.x == 1 || BoundaryIni_f.y== 1|| BoundaryIni_f.z == 1) elements.fixed_boundary_elements.push_back(ElementNumber);

			ElementNumber++;
			ss.clear();
		}
	}

	
	int N = elements.node_num.size();  // number of elements
	elements.boundary_layers.resize(N);

	// mark the boundary layer 1 and 2
	for (int k = 0; k < elements.node_num.size(); k++) {
		elements.boundary_layers[k].x = 0;
		elements.boundary_layers[k].y = 0;
	}

	int NC = elements.cutting_boundary_elements.size();  // number of elements											 
	// Color the mesh of boundary element first (layer 1)
	int ran_num;  // random number
	int na, nb, nc;    // register for storing the node number of each elements
	int cutting_ele_no;
	srand(time(NULL));
	for (int i = 0; i < NC; i++) {
		// generate a random number as the prescribed color

		ran_num = rand() % color_cutting_edge + 1;    
		cutting_ele_no = elements.cutting_boundary_elements[i];
		elements.boundary_layers[cutting_ele_no].x = 1;
		na = elements.node_num[cutting_ele_no].x;
		nb = elements.node_num[cutting_ele_no].y;
		nc = elements.node_num[cutting_ele_no].z;
		int loop_num = 0;
		bool boo = 0;
		while (boo == 0) {
			// find elements sharing the same node
			int neighbor_ele_no;
			for (int j = 0; j < NC; j++) {
				neighbor_ele_no = elements.cutting_boundary_elements[j];
				if (neighbor_ele_no == cutting_ele_no) continue;
				int4 n_buffer = elements.node_num[neighbor_ele_no];
				if (na == n_buffer.x || na == n_buffer.y || na == n_buffer.z
					|| nb == n_buffer.x || nb == n_buffer.y || nb == n_buffer.z
					|| nc == n_buffer.x || nc == n_buffer.y || nc == n_buffer.z) {
					if (elements.node_num[neighbor_ele_no].w == ran_num) { goto change_color_1; }
				}
			}
			boo = 1;
			continue;
			// if the color has been used in the neighbor elements, change the color to the next one
		change_color_1:;
			ran_num = (ran_num + 1);
			if (ran_num > color_cutting_edge) {
				ran_num = 1;
			}
			loop_num += 1;

			assert(loop_num < color_cutting_edge);
		}
		elements.node_num[cutting_ele_no].w = ran_num;
	}
#ifdef WEAR_NODE_SHIFT
	// Mark and color the mesh in the layer 2
	std::vector<int>::iterator ite1;
	std::vector<int>::iterator ite2;
	std::vector<int>::iterator ite3;
	
	for (int kk = 0; kk < N; kk++) {
		if (elements.boundary_layers[kk].x == 1) continue;
		na = elements.node_num[kk].x;
		nb = elements.node_num[kk].y;
		nc = elements.node_num[kk].z;
		ite1 = std::find(CuttingEdgeNum.begin(), CuttingEdgeNum.end(), na);
		ite2 = std::find(CuttingEdgeNum.begin(), CuttingEdgeNum.end(), nb);
		ite3 = std::find(CuttingEdgeNum.begin(), CuttingEdgeNum.end(), nc);
		if (ite1 != CuttingEdgeNum.end() || ite2 != CuttingEdgeNum.end() || ite3 != CuttingEdgeNum.end()) goto color_map;

		int naa, nbb, ncc;
		for (int kkk = 0; kkk < NC; kkk++) {
			int bff = elements.cutting_boundary_elements[kkk];
			naa = elements.node_num[bff].x;
			nbb = elements.node_num[bff].y;
			ncc = elements.node_num[bff].z;
			if (na == naa || na == nbb || na == ncc ||
				nb == naa || nb == nbb || nb == ncc ||
				nc == naa || nc == nbb || nc == ncc
				) {
				goto color_map;
			}
		}
		continue;
	color_map:;
		elements.boundary_layers[kk].y = 1;
		srand(time(NULL));
		ran_num = rand() % color_total + 1;
		int loop_num = 0;
		bool boo = 0;
		while (boo == 0) {
			// find elements sharing the same node
			for (int j = 0; j < N; j++) {
				if (kk == j) continue;
				int4 n_buffer = elements.node_num[j];
				if (na == n_buffer.x || na == n_buffer.y || na == n_buffer.z
					|| nb == n_buffer.x || nb == n_buffer.y || nb == n_buffer.z
					|| nc == n_buffer.x || nc == n_buffer.y || nc == n_buffer.z) {
					if (elements.node_num[j].w == ran_num) { goto change_color_2; }
				}
			}
			boo = 1;
			continue;
			// if the color has been used in the neighbor elements, change the color to the next one
		change_color_2:;
			ran_num = (ran_num + 1);
			if (ran_num > color_total) {
				ran_num = 1;
			}
			loop_num += 1;

			assert(loop_num < color_total);
		}
		elements.node_num[kk].w = ran_num;


	}
#endif
	// Color the remaining elements
	for (int i = 0; i < N; i++) {
		if (elements.node_num[i].w != 0) continue;
		// generate a random number as the prescribed color
		srand(time(NULL));
		ran_num = rand() % color_total + 1;
		na = elements.node_num[i].x;
		nb = elements.node_num[i].y;
		nc = elements.node_num[i].z;
		int loop_num = 0;
		bool boo = 0;
		while (boo == 0) {
			// find elements sharing the same node
			for (int j = 0; j < N; j++) {
				if (i == j) continue;
				int4 n_buffer = elements.node_num[j];
				if (na == n_buffer.x || na == n_buffer.y || na == n_buffer.z
					|| nb == n_buffer.x || nb == n_buffer.y || nb == n_buffer.z
					|| nc == n_buffer.x || nc == n_buffer.y || nc == n_buffer.z) {
					if (elements.node_num[j].w == ran_num) { goto change_color_3; }
				}
			}
			boo = 1;
			continue;
			// if the color has been used in the neighbor elements, change the color to the next one
		    change_color_3:;
			ran_num = (ran_num + 1);
			if (ran_num > color_total ) { 
				ran_num = 1; 
			}
			loop_num += 1;

			assert(loop_num < color_total);
		}
		elements.node_num[i].w = ran_num;
	}
	
	// generate index vectors for CSR matrix
	
	std::vector<int> index_vector;
	int NZ = 0; // number of non zeros
	nodes.row_ptr.resize(NodeNumber + int(1));
	nodes.row_ptr[0] = 0;
	for (int k = 0; k < NodeNumber; k++) {
		int row_num_buffer = 0;
		// find elements containing this node
		for (int m = 0; m < N; m++) {
			if (elements.node_num[m].x == k || elements.node_num[m].y == k || elements.node_num[m].z == k) {
				if (find(index_vector.begin(), index_vector.end(), k * NodeNumber + elements.node_num[m].x) == index_vector.end()) {
					index_vector.push_back(k* NodeNumber + elements.node_num[m].x );
					row_num_buffer += 1;
				}
				if (find(index_vector.begin(), index_vector.end(), k * NodeNumber + elements.node_num[m].y) == index_vector.end()) {
					index_vector.push_back(k * NodeNumber + elements.node_num[m].y);
					row_num_buffer += 1;
				}
				if (find(index_vector.begin(), index_vector.end(), k * NodeNumber + elements.node_num[m].z) == index_vector.end()) {
					index_vector.push_back(k * NodeNumber + elements.node_num[m].z);
					row_num_buffer += 1;
				}
		    } 
		}
		NZ = NZ + row_num_buffer;
		nodes.row_ptr[k + 1] = NZ;
		//nodes.row_ptr[k + 1] = row_num_buffer;
	}

	nodes.col_ind.resize(NZ);
	std::sort(index_vector.begin(), index_vector.end());
	int ll = 0;
	for (int jj = 0; jj < NodeNumber; jj ++) {
		for (int kk = 0; kk < (nodes.row_ptr[jj + 1] - nodes.row_ptr[jj]); kk ++) {
			nodes.col_ind[ll] = index_vector[ll] - jj * NodeNumber;
			ll += 1;
		}
	}
	assert(ll == NZ);

}

mesh_read::mesh_read(std::string file_name)
{
	std::ifstream infile(file_name, std::ifstream::in);

	std::string line;
	std::string NodeSet("Nset");
	std::string ElementSet("Elset");
	std::string NodeID("*Node");
	std::string ElementID("Element");
	std::string CuttingEdgeID("CuttingEdge");
	std::string FixedEdgeID("FixedBoundary");
	std::string StopID("End");
	std::string comment_id("#");
	bool NodeStart = 0;
	bool ElementStart = 0;
	bool CuttingEdgeStart = 0;
	bool FixedEdgeStart = 0;
	bool CuttingEdgeInElements = 0;
	bool FixedEdgeInElements = 0;
	int NodeNumber = 0;
	int ElementNumber = 0;
	//float_t T0 = 300.;
	int3 BoundaryIni; //MeshStructure::tri_boundary BoundaryIni;
	BoundaryIni.x = 0;
	BoundaryIni.y = 0;
	BoundaryIni.z = 0;
	std::vector<float_t> x_;
	std::vector<float_t> y_;
	std::vector<int> CuttingEdgeNum;
	std::vector<int> FixedEdgeNum;
	std::vector<int> CuttingBoundaryCorrection;
	std::vector<int> FixedBoundaryCorrection;
	bool CuttingBoundaryCorrectionBool = 0;
	bool FixedBoundaryCorrectionBool = 0;
	
	while (std::getline(infile, line)) {

		if (line.find(comment_id) != std::string::npos) {
			// npos is a static member constant value with the greatest possible value for an element of type size_t. contains "find" at least once.
			continue;
		}

		if (line.find(NodeID) != std::string::npos) {
			NodeStart = 1;
			continue;
		}

		if (line.find(ElementID) != std::string::npos) {
			ElementStart = 1;
			NodeStart = 0;
			int n = x_.size();
			int zero = 0;
			assert(n == NodeNumber);
			nodes.x_init.resize(n);
            nodes.y_init.resize(n);
			nodes.cutting_edge_flag.resize(n);
			nodes.fixed_edge_flag.resize(n);
			for (int i = 0; i < n; i++) {
				nodes.x_init[i] = x_[i] / 100 ;
				nodes.y_init[i] = y_[i] / 100;
				nodes.cutting_edge_flag[i] = zero;
				nodes.fixed_edge_flag[i] = zero;
			}

			continue;
		}

		if (line.find(CuttingEdgeID) != std::string::npos && line.find(NodeSet) != std::string::npos) {
			CuttingEdgeStart = 1;
			ElementStart = 0;
			continue;
		}

		if (line.find(CuttingEdgeID) != std::string::npos && line.find(ElementSet) != std::string::npos) {
			CuttingEdgeInElements = 1;
			CuttingEdgeStart = 0;
			//std::cout << "Register cutting boundary nodes successfully!" << std::endl;
			continue;
		}

		if (line.find(FixedEdgeID) != std::string::npos && line.find(NodeSet) != std::string::npos) {
			FixedEdgeStart = 1;
			CuttingEdgeInElements = 0;
			if (CuttingBoundaryCorrectionBool) {
				for (int a : CuttingBoundaryCorrection) {

					for (int b : elements.cutting_boundary_elements) {
						if (b != a) {
							if (elements.node_num[a].x == elements.node_num[b].x || elements.node_num[a].x == elements.node_num[b].y || elements.node_num[a].x == elements.node_num[b].z) { goto a1; }
						}
					}
					elements.cutting_boundary_flags[a].y = 0;
				a1:;
					for (int b : elements.cutting_boundary_elements) {
						if (b != a) {
							if (elements.node_num[a].y == elements.node_num[b].x || elements.node_num[a].y == elements.node_num[b].y || elements.node_num[a].y == elements.node_num[b].z) { goto a2; }
						}
					}
					elements.cutting_boundary_flags[a].z = 0;
				a2:;
					for (int b : elements.cutting_boundary_elements) {
						if (b != a) {
							if (elements.node_num[a].z == elements.node_num[b].x || elements.node_num[a].z == elements.node_num[b].y || elements.node_num[a].z == elements.node_num[b].z) { goto a3; }
						}
					}
					elements.cutting_boundary_flags[a].x = 0;
				a3:;
				}
				
			}
			continue;
		}

		if (line.find(FixedEdgeID) != std::string::npos && line.find(ElementSet) != std::string::npos) {
			FixedEdgeInElements = 1;
			FixedEdgeStart = 0;
			continue;
		}

		if (line.find(StopID) != std::string::npos) {
			if (FixedBoundaryCorrectionBool) {
				for (int a : FixedBoundaryCorrection) {

					for (int b : elements.fixed_boundary_elements) {
						if (b != a) {
							if (elements.node_num[a].x == elements.node_num[b].x || elements.node_num[a].x == elements.node_num[b].y || elements.node_num[a].x == elements.node_num[b].z) { goto b1; }
						}
					}
					elements.fixed_boundary_flags[a].y = 0;
				b1:;
					for (int b : elements.fixed_boundary_elements) {
						if (b != a) {
							if (elements.node_num[a].y == elements.node_num[b].x || elements.node_num[a].y == elements.node_num[b].y || elements.node_num[a].y == elements.node_num[b].z) { goto b2; }
						}
					}
					elements.fixed_boundary_flags[a].z = 0;
				b2:;
					for (int b : elements.fixed_boundary_elements) {
						if (b != a) {
							if (elements.node_num[a].z == elements.node_num[b].x || elements.node_num[a].z == elements.node_num[b].y || elements.node_num[a].z == elements.node_num[b].z) { goto b3; }
						}
					}
					elements.fixed_boundary_flags[a].x = 0;
				b3:;
				}

			}
			FixedEdgeInElements = 0;
			break;
		}

		if (NodeStart) {
			std::istringstream ss(line);
			int a;
			float_t x;
			float_t y;
			std::size_t offset = 0;
			std::string token;
			std::getline(ss, token, ',');
			std::stringstream nu(token);
			nu >> a;
			std::getline(ss, token, ',');
			x = std::stod(token, &offset);
			std::getline(ss, token);
			y = std::stod(token, &offset);
			x_.push_back(x);
			y_.push_back(y);
			NodeNumber++;
			ss.clear();
		}

		if (ElementStart) {
			std::istringstream ss(line);
			std::string token;
			std::vector<std::string> buf;
			std::vector<int> bufint;
			int a;
			while (std::getline(ss, token, ',')) {
				buf.push_back(token);
			}
			int4 buff;
			for (int i = 1; i < 4; i++) {
				std::stringstream nu(buf[i]);
				nu >> a;
				bufint.push_back(a - 1);
			}
			buff.x = bufint[0];
			buff.y = bufint[1];
			buff.z = bufint[2];
			buff.w = 0;
			elements.node_num.push_back(buff);
			elements.cutting_boundary_flags.push_back(BoundaryIni);
			elements.fixed_boundary_flags.push_back(BoundaryIni);
			ElementNumber++;
			ss.clear();
		}

		if (CuttingEdgeStart) {
			std::istringstream ss(line);
			int a;
			std::string token;
			while (std::getline(ss, token, ',')) {
				std::stringstream nu(token);
				nu >> a;
				nodes.cutting_edge_flag[a - 1] = 1;
				CuttingEdgeNum.push_back(a - 1);
			}
			ss.clear();
		}

		if (CuttingEdgeInElements) {
			std::istringstream ss(line);
			int a;
			std::string token;
			int4 BufferElement;
			std::vector<int>::iterator it1;
			std::vector<int>::iterator it2;
			std::vector<int>::iterator it3;
			while (std::getline(ss, token, ',')) {
				std::stringstream nu(token);
				nu >> a;
				elements.cutting_boundary_elements.push_back(a - 1);
				elements.cutting_boundary_contact.push_back(0);
				BufferElement = elements.node_num[a - 1];
				it1 = std::find(CuttingEdgeNum.begin(), CuttingEdgeNum.end(), BufferElement.x);
				it2 = std::find(CuttingEdgeNum.begin(), CuttingEdgeNum.end(), BufferElement.y);
				it3 = std::find(CuttingEdgeNum.begin(), CuttingEdgeNum.end(), BufferElement.z);
				if (it1 != CuttingEdgeNum.end() && it2 != CuttingEdgeNum.end()) { elements.cutting_boundary_flags[a - 1].x = 1; }
				if (it2 != CuttingEdgeNum.end() && it3 != CuttingEdgeNum.end()) { elements.cutting_boundary_flags[a - 1].y = 1; }
				if (it3 != CuttingEdgeNum.end() && it1 != CuttingEdgeNum.end()) { elements.cutting_boundary_flags[a - 1].z = 1; }
				if (elements.cutting_boundary_flags[a - 1].x == 1 && elements.cutting_boundary_flags[a - 1].y == 1 && elements.cutting_boundary_flags[a - 1].z == 1) {
					CuttingBoundaryCorrectionBool = 1;
					CuttingBoundaryCorrection.push_back(a - 1);
				}
			}
			ss.clear();
		}

		if (FixedEdgeStart) {
			std::istringstream ss(line);
			int a;
			std::string token;
			while (std::getline(ss, token, ',')) {
				std::stringstream nu(token);
				nu >> a;
				nodes.fixed_edge_flag[a - 1] = 1;
				FixedEdgeNum.push_back(a - 1);
			}
			ss.clear();
		}

		if (FixedEdgeInElements) {
			std::istringstream ss(line);
			int a;
			std::string token;
			int4 BufferElement;
			std::vector<int>::iterator it1;
			std::vector<int>::iterator it2;
			std::vector<int>::iterator it3;
			while (std::getline(ss, token, ',')) {
				std::stringstream nu(token);
				nu >> a;
				elements.fixed_boundary_elements.push_back(a - 1);
				BufferElement = elements.node_num[a - 1];
				it1 = std::find(FixedEdgeNum.begin(), FixedEdgeNum.end(), BufferElement.x);
				it2 = std::find(FixedEdgeNum.begin(), FixedEdgeNum.end(), BufferElement.y);
				it3 = std::find(FixedEdgeNum.begin(), FixedEdgeNum.end(), BufferElement.z);
				if (it1 != FixedEdgeNum.end() && it2 != FixedEdgeNum.end()) { elements.fixed_boundary_flags[a - 1].x = 1; }
				if (it2 != FixedEdgeNum.end() && it3 != FixedEdgeNum.end()) { elements.fixed_boundary_flags[a - 1].y = 1; }
				if (it3 != FixedEdgeNum.end() && it1 != FixedEdgeNum.end()) { elements.fixed_boundary_flags[a - 1].z = 1; }
				if (elements.fixed_boundary_flags[a - 1].x == 1 && elements.fixed_boundary_flags[a - 1].y == 1 && elements.fixed_boundary_flags[a - 1].z == 1) {
					FixedBoundaryCorrectionBool = 1;
					FixedBoundaryCorrection.push_back(a - 1);
				}
			}
			ss.clear();
		}

	}

	int N = elements.node_num.size();  // number of elements
	elements.boundary_layers.resize(N);
	// mark the boundary layer 1 and 2
	for (int k = 0; k < elements.node_num.size(); k++) {
		elements.boundary_layers[k].x = 0;
		elements.boundary_layers[k].y = 0;
	}

	int NC = elements.cutting_boundary_elements.size();  // number of elements
														 
	// Color the mesh of boundary element first (layer 1)
	int ran_num;  // random number
	int na, nb, nc;    // register for storing the node number of each elements
	int cutting_ele_no;
	srand(time(NULL));
	for (int i = 0; i < NC; i++) {
		// generate a random number as the prescribed color

		ran_num = rand() % color_cutting_edge + 1;    
		cutting_ele_no = elements.cutting_boundary_elements[i];
		elements.boundary_layers[cutting_ele_no].x = 1;
		na = elements.node_num[cutting_ele_no].x;
		nb = elements.node_num[cutting_ele_no].y;
		nc = elements.node_num[cutting_ele_no].z;
		int loop_num = 0;
		bool boo = 0;
		while (boo == 0) {
			// find elements sharing the same node
			int neighbor_ele_no;
			for (int j = 0; j < NC; j++) {
				neighbor_ele_no = elements.cutting_boundary_elements[j];
				if (neighbor_ele_no == cutting_ele_no) continue;
				int4 n_buffer = elements.node_num[neighbor_ele_no];
				if (na == n_buffer.x || na == n_buffer.y || na == n_buffer.z
					|| nb == n_buffer.x || nb == n_buffer.y || nb == n_buffer.z
					|| nc == n_buffer.x || nc == n_buffer.y || nc == n_buffer.z) {
					if (elements.node_num[neighbor_ele_no].w == ran_num) { goto change_color_1; }
				}
			}
			boo = 1;
			continue;
			// if the color has been used in the neighbor elements, change the color to the next one
		change_color_1:;
			ran_num = (ran_num + 1);
			if (ran_num > color_cutting_edge) {
				ran_num = 1;
			}
			loop_num += 1;

			assert(loop_num < color_cutting_edge);
		}
		elements.node_num[cutting_ele_no].w = ran_num;
	}
#ifdef WEAR_NODE_SHIFT
	// Mark and color the mesh in the layer 2
	std::vector<int>::iterator ite1;
	std::vector<int>::iterator ite2;
	std::vector<int>::iterator ite3;
	
	for (int kk = 0; kk < N; kk++) {
		if (elements.boundary_layers[kk].x == 1) continue;
		na = elements.node_num[kk].x;
		nb = elements.node_num[kk].y;
		nc = elements.node_num[kk].z;
		ite1 = std::find(CuttingEdgeNum.begin(), CuttingEdgeNum.end(), na);
		ite2 = std::find(CuttingEdgeNum.begin(), CuttingEdgeNum.end(), nb);
		ite3 = std::find(CuttingEdgeNum.begin(), CuttingEdgeNum.end(), nc);
		if (ite1 != CuttingEdgeNum.end() || ite2 != CuttingEdgeNum.end() || ite3 != CuttingEdgeNum.end()) goto color_map;

		int naa, nbb, ncc;
		for (int kkk = 0; kkk < NC; kkk++) {
			int bff = elements.cutting_boundary_elements[kkk];
			naa = elements.node_num[bff].x;
			nbb = elements.node_num[bff].y;
			ncc = elements.node_num[bff].z;
			if (na == naa || na == nbb || na == ncc ||
				nb == naa || nb == nbb || nb == ncc ||
				nc == naa || nc == nbb || nc == ncc
				) {
				goto color_map;
			}
		}
		continue;
	color_map:;
		elements.boundary_layers[kk].y = 1;
		srand(time(NULL));
		ran_num = rand() % color_total + 1;
		int loop_num = 0;
		bool boo = 0;
		while (boo == 0) {
			// find elements sharing the same node
			for (int j = 0; j < N; j++) {
				if (kk == j) continue;
				int4 n_buffer = elements.node_num[j];
				if (na == n_buffer.x || na == n_buffer.y || na == n_buffer.z
					|| nb == n_buffer.x || nb == n_buffer.y || nb == n_buffer.z
					|| nc == n_buffer.x || nc == n_buffer.y || nc == n_buffer.z) {
					if (elements.node_num[j].w == ran_num) { goto change_color_2; }
				}
			}
			boo = 1;
			continue;
			// if the color has been used in the neighbor elements, change the color to the next one
		change_color_2:;
			ran_num = (ran_num + 1);
			if (ran_num > color_total) {
				ran_num = 1;
			}
			loop_num += 1;

			assert(loop_num < color_total);
		}
		elements.node_num[kk].w = ran_num;


	}
#endif
	// Color the remaining elements
	for (int i = 0; i < N; i++) {
		if (elements.node_num[i].w != 0) continue;
		// generate a random number as the prescribed color
		srand(time(NULL));
		ran_num = rand() % color_total + 1;
		na = elements.node_num[i].x;
		nb = elements.node_num[i].y;
		nc = elements.node_num[i].z;
		int loop_num = 0;
		bool boo = 0;
		while (boo == 0) {
			// find elements sharing the same node
			for (int j = 0; j < N; j++) {
				if (i == j) continue;
				int4 n_buffer = elements.node_num[j];
				if (na == n_buffer.x || na == n_buffer.y || na == n_buffer.z
					|| nb == n_buffer.x || nb == n_buffer.y || nb == n_buffer.z
					|| nc == n_buffer.x || nc == n_buffer.y || nc == n_buffer.z) {
					if (elements.node_num[j].w == ran_num) { goto change_color_3; }
				}
			}
			boo = 1;
			continue;
			// if the color has been used in the neighbor elements, change the color to the next one
		    change_color_3:;
			ran_num = (ran_num + 1);
			if (ran_num > color_total ) { 
				ran_num = 1; 
			}
			loop_num += 1;

			assert(loop_num < color_total);
		}
		elements.node_num[i].w = ran_num;
	}
	
	// generate index vectors for CSR matrix
	
	std::vector<int> index_vector;
	int NZ = 0; // number of non zeros
	nodes.row_ptr.resize(NodeNumber + int(1));
	nodes.row_ptr[0] = 0;
	for (int k = 0; k < NodeNumber; k++) {
		int row_num_buffer = 0;
		// find elements containing this node
		for (int m = 0; m < N; m++) {
			if (elements.node_num[m].x == k || elements.node_num[m].y == k || elements.node_num[m].z == k) {
				if (find(index_vector.begin(), index_vector.end(), k * NodeNumber + elements.node_num[m].x) == index_vector.end()) {
					index_vector.push_back(k* NodeNumber + elements.node_num[m].x );
					row_num_buffer += 1;
				}
				if (find(index_vector.begin(), index_vector.end(), k * NodeNumber + elements.node_num[m].y) == index_vector.end()) {
					index_vector.push_back(k * NodeNumber + elements.node_num[m].y);
					row_num_buffer += 1;
				}
				if (find(index_vector.begin(), index_vector.end(), k * NodeNumber + elements.node_num[m].z) == index_vector.end()) {
					index_vector.push_back(k * NodeNumber + elements.node_num[m].z);
					row_num_buffer += 1;
				}
		    }
		}
		NZ = NZ + row_num_buffer;
		nodes.row_ptr[k + 1] = NZ;
	}
	nodes.col_ind.resize(NZ);
	std::sort(index_vector.begin(), index_vector.end());
	int ll = 0;
	for (int jj = 0; jj < NodeNumber; jj ++) {
		for (int kk = 0; kk < (nodes.row_ptr[jj + 1] - nodes.row_ptr[jj]); kk ++) {
			nodes.col_ind[ll] = index_vector[ll] - jj * NodeNumber;
			ll += 1;
		}
	}
	assert(ll == NZ);

}

void mesh_read::mesh_read_temperature(std::string file_name)
{
#ifdef USE_HOT_TOOL
	std::ifstream infile(file_name, std::ifstream::in);
	std::string line;
	std::string StartId("LOOKUP_TABLE default");
	std::vector<float_t> T_r;
	bool temp_read_start = false;
	while (std::getline(infile, line)) {

		if (line.find(StartId) != std::string::npos) {
			temp_read_start = true;
			continue;
		}


		if (temp_read_start) {
			std::istringstream ss(line);
			float_t T_read;
			std::size_t offset = 0;
			std::string token;
			std::getline(ss, token);
			if (token.size() == 0)    continue;          /* if .size() == 0, empty line */
			T_read = std::stod(token, &offset);
			T_r.push_back(T_read);
			ss.clear();
		}
	}
	int n = T_r.size();
	int N = nodes.x_init.size();
	assert(n == N);
	nodes.T_init.resize(n);
	for (int i = 0; i < n; i++) {
		nodes.T_init[i] = T_r[i];
	}
	printf("Read temperature data successfully!\n");
#endif
}

void mesh_GPU_m::reset_mesh_gpu(mesh_GPU_m* mesh_GPU_new){
    ele_num_total = mesh_GPU_new->get_element_number();
	ele_num_cutting = mesh_GPU_new->get_cutting_edge_element_number();
    ele_num_fixed = mesh_GPU_new->get_fixed_edge_element_number();
	no_num = mesh_GPU_new->get_node_number();
	non_zeros = mesh_GPU_new->get_non_zero_number();
	color_total = mesh_GPU_new->get_color_number();
	cutting_edge_color = mesh_GPU_new->get_cutting_edge_color_number();

	int num_node = get_node_number();
	int num_tri = get_element_number();
	int non_zeros = get_non_zero_number();
	

	cudaFree(position);
	cudaFree(cutting_edge_flag);
	cudaFree(fixed_edge_flag);
	cudaFree(T_nodes);
	cudaFree(f);
	cudaFree(k);
#ifdef WEAR_NODE_SHIFT
	cudaFree(position_mapping);
#endif	

    cudaFree(node_num);
    cudaFree(cutting_boundary_flags);
	cudaFree(fixed_boundary_flags);
#ifdef WEAR_NODE_SHIFT
	cudaFree(cutting_boundary_layer);
	cudaFree(node_num_mapping);
#endif
	cudaFree(cutting_boundary_elements);
	cudaFree(fixed_boundary_elements);


	cudaFree(csr_row_ptr);
	cudaFree(csr_col_ind);

	cudaMalloc((void **) &position, sizeof(float2_t)*num_node);
    cudaMalloc((void **) &cutting_edge_flag, sizeof(int)*num_node);
    cudaMalloc((void **) &fixed_edge_flag, sizeof(int)*num_node);
    cudaMalloc((void **) &T_nodes, sizeof(float_t)*num_node);
    cudaMalloc((void **) &f, sizeof(float_t)*num_node);
    cudaMalloc((void **) &k, sizeof(float_t)*num_node);
	
#ifdef WEAR_NODE_SHIFT
	cudaMalloc((void**) &position_mapping, sizeof(float2_t) * num_node);
#endif

    cudaMalloc((void **) &node_num, sizeof(int4) * num_tri);
    cudaMalloc((void **) &cutting_boundary_flags, sizeof(int3) * num_tri);
    cudaMalloc((void **) &fixed_boundary_flags, sizeof(int3) * num_tri);
#ifdef WEAR_NODE_SHIFT
	cudaMalloc((void**) &cutting_boundary_layer, sizeof(int2) * num_tri);
	cudaMalloc((void**) &node_num_mapping, sizeof(int4) * num_tri);
#endif
    cudaMalloc((void **) &cutting_boundary_elements, sizeof(int) * ele_num_cutting);
    cudaMalloc((void **) &fixed_boundary_elements, sizeof(int) * ele_num_fixed);

	cudaMalloc((void **) &csr_row_ptr, sizeof(int) * (num_node + 1));
	cudaMalloc((void **) &csr_col_ind, sizeof(int) * non_zeros);


	cudaMemcpy(position, mesh_GPU_new->position, sizeof(float2_t) * (num_node), cudaMemcpyDeviceToDevice);
	cudaMemcpy(cutting_edge_flag, mesh_GPU_new->cutting_edge_flag, sizeof(int) * (num_node), cudaMemcpyDeviceToDevice);
	cudaMemcpy(fixed_edge_flag, mesh_GPU_new->fixed_edge_flag, sizeof(int) * (num_node), cudaMemcpyDeviceToDevice);
	cudaMemcpy(T_nodes, mesh_GPU_new->T_nodes, sizeof(float_t) * (num_node), cudaMemcpyDeviceToDevice);
	cudaMemcpy(f, mesh_GPU_new->f, sizeof(float_t) * (num_node), cudaMemcpyDeviceToDevice);
	cudaMemcpy(k, mesh_GPU_new->k, sizeof(float_t) * (num_node), cudaMemcpyDeviceToDevice);
#ifdef WEAR_NODE_SHIFT
	cudaMemcpy(position_mapping, mesh_GPU_new->position_mapping, sizeof(float2_t) * (num_node), cudaMemcpyDeviceToDevice);
#endif

    

	cudaMemcpy(node_num, mesh_GPU_new->node_num, sizeof(int4) * (num_tri), cudaMemcpyDeviceToDevice);
	cudaMemcpy(cutting_boundary_flags, mesh_GPU_new->cutting_boundary_flags, sizeof(int3) * (num_tri), cudaMemcpyDeviceToDevice);
	cudaMemcpy(fixed_boundary_flags, mesh_GPU_new->fixed_boundary_flags, sizeof(int3) * (num_tri), cudaMemcpyDeviceToDevice);

#ifdef WEAR_NODE_SHIFT
	cudaMemcpy(cutting_boundary_layer, mesh_GPU_new->cutting_boundary_layer, sizeof(int2) * (num_tri), cudaMemcpyDeviceToDevice);
	cudaMemcpy(node_num_mapping, mesh_GPU_new->node_num, sizeof(int4) * (num_tri), cudaMemcpyDeviceToDevice);
#endif

	cudaMemcpy(cutting_boundary_elements, mesh_GPU_new->cutting_boundary_elements, sizeof(int) * (ele_num_cutting), cudaMemcpyDeviceToDevice);
	cudaMemcpy(fixed_boundary_elements, mesh_GPU_new->fixed_boundary_elements, sizeof(int) * (ele_num_fixed), cudaMemcpyDeviceToDevice);

	cudaMemcpy(csr_row_ptr, mesh_GPU_new->csr_row_ptr, sizeof(int) * (no_num + 1), cudaMemcpyDeviceToDevice);
	cudaMemcpy(csr_col_ind, mesh_GPU_new->csr_col_ind, sizeof(int) * non_zeros, cudaMemcpyDeviceToDevice);
	
	check_cuda_error("Remesh_set_gpu_new\n");
}

mesh_read::mesh_read()
{
}

void mesh_read::write_mesh_file()
{
	struct triangle {
		glm::dvec2 p1, p2, p3;
		triangle(glm::dvec2 p1, glm::dvec2 p2, glm::dvec2 p3) : p1(p1), p2(p2), p3(p3) {}
	};

	std::vector<triangle> triangles;

	int num_node = nodes.x_init.size();
	int num_tri = elements.node_num.size();

	char buf[256];
	sprintf(buf, "Tool_mesh.vtk");
	FILE* fp = fopen(buf, "w+");

	fprintf(fp, "# vtk DataFile Version 2.0\n");
	fprintf(fp, "mfree iwf\n");
	fprintf(fp, "ASCII\n");
	fprintf(fp, "\n");
	fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
	fprintf(fp, "POINTS %d float\n", num_node);

	double x1, y1;
	for (int i = 0; i < num_node; i++) {
		x1 = nodes.x_init[i];
		y1 = nodes.y_init[i];
		fprintf(fp, "%f %f %f\n", x1, y1, 0.);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELLS %d %d\n", num_tri, 3 * num_tri + num_tri);
	for (int i = 0; i < num_tri; i++) {
		fprintf(fp, "3 %d %d %d\n", elements.node_num[i].x, elements.node_num[i].y, elements.node_num[i].z);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELL_TYPES %d\n", num_tri);
	for (int i = 0; i < num_tri; i++) {
		fprintf(fp, "5\n");
	}

	fprintf(fp, "CELL_DATA %d\n", num_tri);

	fprintf(fp, "SCALARS color int 1\n");		
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (int i = 0; i < num_tri; i++) {
		fprintf(fp, "%u\n", elements.node_num[i].w);
	}
	fprintf(fp, "\n");

	fprintf(fp, "SCALARS layerf int 1\n");		
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (int i = 0; i < num_tri; i++) {
		fprintf(fp, "%u\n", elements.boundary_layers[i].x);
	}
	fprintf(fp, "\n");

	fprintf(fp, "SCALARS layers int 1\n");		
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (int i = 0; i < num_tri; i++) {
		fprintf(fp, "%u\n", elements.boundary_layers[i].y);
	}
	fprintf(fp, "\n");

	fclose(fp);
}


void generate_tool_mesh_gmsh(std::vector<vec2_t> points, float_t r, float_t lc)
{
    float2_t tl = make_float2_t(points[0].x, points[0].y);
	float2_t tr = make_float2_t(points[1].x, points[1].y);
	float2_t np1 = make_float2_t(points[2].x, points[2].y);
	float2_t np2 = make_float2_t(points[3].x, points[3].y);
	float2_t bl = make_float2_t(points[4].x, points[4].y);

	float_t rad = r;

	float_t len_half = 0.5 * pow(((np1.x - np2.x) *(np1.x - np2.x) + (np1.y - np2.y)*(np1.y-np2.y)), 0.5);
	float_t length_perp = pow((rad * rad - len_half * len_half), 0.5);

	float2_t tc = make_float2_t((0.5 * (np1.x + np2.x) - length_perp * (np1.y - np2.y) / len_half / 2.),(0.5 * (np1.y + np2.y) + length_perp * (np1.x - np2.x) / len_half / 2.));


	gmsh::initialize();
	gmsh::option::setNumber("General.Verbosity", 2);
	gmsh::model::add("tool");

	gmsh::model::geo::addPoint(tl.x, tl.y, 0, 0.07, 1);
	gmsh::model::geo::addPoint(tr.x, tr.y, 0, 0.003, 2);
	gmsh::model::geo::addPoint(np1.x, np1.y, 0, lc, 3);
	gmsh::model::geo::addPoint(np2.x, np2.y, 0, lc, 4);
	gmsh::model::geo::addPoint(bl.x, bl.y, 0, 0.003, 5);
	gmsh::model::geo::addPoint(tc.x, tc.y, 0, lc, 6);


	gmsh::model::geo::addLine(1, 2, 1);
    gmsh::model::geo::addLine(2, 3, 2);
    gmsh::model::geo::addCircleArc(3, 6, 4);
    gmsh::model::geo::addLine(4, 5, 4);
    gmsh::model::geo::addLine(5, 1, 5);

	gmsh::model::geo::addCurveLoop({1,2,3,4,5}, 1);

	gmsh::model::geo::addPlaneSurface({1}, 1);

	gmsh::model::geo::synchronize();

	gmsh::model::addPhysicalGroup(1, {2,3,4}, 67, "CuttingEdge");
    gmsh::model::addPhysicalGroup(1, {1,5}, 2, "FixedBoundary");
	gmsh::model::geo::synchronize();

	gmsh::model::mesh::generate();
    gmsh::option::setNumber("Mesh.MshFileVersion", 4);
	gmsh::option::setNumber("Mesh.SaveAll", 1);

	gmsh::write("tool_gmsh.inp");

	gmsh::finalize();

}