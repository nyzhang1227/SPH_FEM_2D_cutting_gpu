// Copyright ETH Zurich, IWF
// SPH-FEM cutting simulations on the basis of mfree_iwf-ul_cut_gpu: 
// https://github.com/iwf-inspire/mfree_iwf-ul_cut_gpu
// Date: 2024.03.10

// You should have received a copy of the GNU General Public License
// along with mfree_iwf-ul_cut_gpu.  If not, see <http://www.gnu.org/licenses/>.

#include "vtk_writer.h"

void vtk_writer_write(const particle_gpu* particles, int step) {

	static int* h_idx = 0;
	static float2_t* h_pos = 0;
	static float2_t* h_vel = 0;
	static float_t* h_rho = 0;
	static float_t* h_h = 0;
	static float_t* h_p = 0;
	static float_t* h_T = 0;
	static float_t* h_eps = 0;

	static float4_t* h_S = 0;

	static float_t* h_fixed = 0;
	static float_t* h_blanked = 0;
	//static float_t *h_tool_p	= 0;
	static int* h_on_seg = 0;

	static float2_t* h_fc = 0;
	static float2_t* h_ft = 0;
	static float2_t* h_nor = 0;

	if (h_idx == 0) {
		int n_init = particles->N;

		// Memory allocation only upon first call;
		h_idx = new int[n_init];
		h_pos = new float2_t[n_init];
		h_vel = new float2_t[n_init];
		h_rho = new float_t[n_init];
		h_h = new float_t[n_init];
		h_p = new float_t[n_init];
		h_T = new float_t[n_init];
		h_eps = new float_t[n_init];

		h_S = new float4_t[n_init];

		h_fixed = new float_t[n_init];		// BC-particles
		h_blanked = new float_t[n_init];		// blanked particles
		//h_tool_p	= new float_t[n_init];
		h_on_seg = new int[n_init];
		h_fc = new float2_t[n_init];
		h_ft = new float2_t[n_init];
		h_nor = new float2_t[n_init];
	}

	int n = particles->N;

	cudaMemcpy(h_idx, particles->idx, sizeof(int) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_pos, particles->pos, sizeof(float2_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vel, particles->vel, sizeof(float2_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_rho, particles->rho, sizeof(float_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_h, particles->h, sizeof(float_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_p, particles->p, sizeof(float_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_T, particles->T, sizeof(float_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_eps, particles->eps_pl, sizeof(float_t) * n, cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_blanked, particles->blanked, sizeof(float_t)*n,  cudaMemcpyDeviceToHost);
	cudaMemcpy(h_S, particles->S, sizeof(float4_t) * n, cudaMemcpyDeviceToHost);

	cudaMemcpy(h_fixed, particles->fixed, sizeof(float_t) * n, cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_tool_p, particles->tool_particle, sizeof(float_t)*n,  cudaMemcpyDeviceToHost);
	cudaMemcpy(h_on_seg, particles->on_seg, sizeof(int) * n, cudaMemcpyDeviceToHost);

	cudaMemcpy(h_fc, particles->fc, sizeof(float2_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_ft, particles->ft, sizeof(float2_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_nor, particles->n, sizeof(float2_t) * n, cudaMemcpyDeviceToHost);

	int num_unblanked_part = 0;
	for (int i = 0; i < n; i++) {
		if (h_blanked[i] != 1.) {
			num_unblanked_part++;
		}
	}

	char buf[256];
	sprintf(buf, "results/vtk_out_%06d.vtk", step);
	FILE* fp = fopen(buf, "w+");

	fprintf(fp, "# vtk DataFile Version 2.0\n");
	fprintf(fp, "mfree iwf\n");
	fprintf(fp, "ASCII\n");
	fprintf(fp, "\n");

	fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
	fprintf(fp, "POINTS %d float\n", num_unblanked_part);
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i] == 1.) continue;
		fprintf(fp, "%f %f %f\n", h_pos[i].x, h_pos[i].y, 0.);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELLS %d %d\n", num_unblanked_part, 2 * num_unblanked_part);
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i] == 1.) continue;
		fprintf(fp, "%d %d\n", 1, i);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELL_TYPES %d\n", num_unblanked_part);
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i] == 1.) continue;
		fprintf(fp, "%d\n", 1);
	}
	fprintf(fp, "\n");

	fprintf(fp, "POINT_DATA %d\n", num_unblanked_part);
/*	
		fprintf(fp, "SCALARS density float 1\n");
		fprintf(fp, "LOOKUP_TABLE default\n");
		for (unsigned int i = 0; i < n; i++) {
			if (h_blanked[i]==1.) continue;
			fprintf(fp, "%f\n", h_rho[i]);
		}
		fprintf(fp, "\n");
*/		
	fprintf(fp, "SCALARS Temperature float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i] == 1.) continue;
		fprintf(fp, "%f\n", h_T[i]);
	}
	fprintf(fp, "\n");

	/*
		fprintf(fp, "SCALARS Fixed float 1\n");
		fprintf(fp, "LOOKUP_TABLE default\n");
		for (unsigned int i = 0; i < n; i++) {
			if (h_blanked[i]==1.) continue;
			fprintf(fp, "%f\n", h_fixed[i]);
		}
		fprintf(fp, "\n");
	*/

	fprintf(fp, "SCALARS on_seg int 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i] == 1.) continue;
		fprintf(fp, "%d\n", h_on_seg[i]);
	}
	fprintf(fp, "\n");

	/*
		fprintf(fp, "SCALARS Tool float 1\n");
		fprintf(fp, "LOOKUP_TABLE default\n");
		for (unsigned int i = 0; i < n; i++) {
			if (h_blanked[i]==1.) continue;
			fprintf(fp, "%f\n", h_tool_p[i]);
		}
		fprintf(fp, "\n");
	*/
	fprintf(fp, "SCALARS EquivAccumPlasticStrain float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i] == 1.) continue;
		fprintf(fp, "%f\n", h_eps[i]);
	}
	fprintf(fp, "\n");

	fprintf(fp, "VECTORS Velocity float\n");
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i] == 1.) continue;
		fprintf(fp, "%f %f %f\n", h_vel[i].x, h_vel[i].y, 0.);
	}
	fprintf(fp, "\n");

	fprintf(fp, "VECTORS fc float\n");
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i] == 1.) continue;
		fprintf(fp, "%2f %2f %2f\n", 100000 * h_fc[i].x, 100000 * h_fc[i].y, 0.);
	}
	fprintf(fp, "\n");

	fprintf(fp, "VECTORS ft float\n");
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i] == 1.) continue;
		fprintf(fp, "%f %f %f\n", 100000 * h_ft[i].x, 100000 * h_ft[i].y, 0.);
	}
	fprintf(fp, "\n");

	fprintf(fp, "VECTORS normal float\n");
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i] == 1.) continue;
		fprintf(fp, "%f %f %f\n", h_nor[i].x, h_nor[i].y, 0.);
	}
	fprintf(fp, "\n");

	fprintf(fp, "SCALARS SvM float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i] == 1.) continue;

		float_t cxx = h_S[i].x - h_p[i];
		float_t cyy = h_S[i].z - h_p[i];
		float_t czz = h_S[i].w - h_p[i];
		float_t cxy = h_S[i].y;

		float_t svm2 = (cxx * cxx + cyy * cyy + czz * czz) - cxx * cyy - cxx * czz - cyy * czz + 3.0 * cxy * cxy;

		float_t svm = (svm2 > 0) ? sqrt(svm2) : 0.;
		fprintf(fp, "%f\n", svm);
	}
	fprintf(fp, "\n");

	fclose(fp);
}

struct triangle {
	vec2_t p1, p2, p3;
	triangle(vec2_t p1, vec2_t p2, vec2_t p3) : p1(p1), p2(p2), p3(p3) {}
};

void vtk_writer_write(const tool* tool, int step) {
	auto segments = tool->get_segments();

	if (segments.size() == 0) {
		return;
	}

	assert(segments.size() == 4 || segments.size() == 5);

	std::vector<triangle> triangles;

	//mesh tool "body"
	if (segments.size() == 4) {
		triangles.push_back(triangle(segments[0].left, segments[0].right, segments[1].right));
		triangles.push_back(triangle(segments[2].left, segments[2].right, segments[3].right));
	}
	else if (segments.size() == 5) {
		triangles.push_back(triangle(segments[0].left, segments[0].right, segments[2].right));
		triangles.push_back(triangle(segments[1].left, segments[1].right, segments[2].right));
		triangles.push_back(triangle(segments[3].left, segments[3].right, segments[4].right));
	}

	//mesh fillet
	if (tool->get_fillet() != 0) {
		const int num_discr = 20;
		auto fillet = tool->get_fillet();
		float_t t1 = fmin(fillet->t1, fillet->t2);
		float_t t2 = fmax(fillet->t1, fillet->t2);

		float_t lo = t1 - 0.1 * t1;
		//float_t hi = t2 + 0.1*t2;

		float_t d_angle = (t2 - t1) / (num_discr - 1);

		float_t r = fillet->r;

		for (int i = 0; i < num_discr - 1; i++) {
			float_t angle_1 = lo + (i + 0) * d_angle;
			float_t angle_2 = lo + (i + 1) * d_angle;

			vec2_t p1 = vec2_t(fillet->p.x, fillet->p.y);
			vec2_t p2 = vec2_t(p1.x + r * sin(angle_1), p1.y + r * cos(angle_1));
			vec2_t p3 = vec2_t(p1.x + r * sin(angle_2), p1.y + r * cos(angle_2));
			triangles.push_back(triangle(p1, p2, p3));
		}
	}

	int num_tri = triangles.size();

	char buf[256];
	sprintf(buf, "results/vtk_tool_%06d.vtk", step);
	FILE* fp = fopen(buf, "w+");

	fprintf(fp, "# vtk DataFile Version 2.0\n");
	fprintf(fp, "mfree iwf\n");
	fprintf(fp, "ASCII\n");
	fprintf(fp, "\n");
	fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
	fprintf(fp, "POINTS %d float\n", 3 * num_tri);

	for (auto it : triangles) {
		fprintf(fp, "%f %f %f\n", it.p1.x, it.p1.y, 0.);
		fprintf(fp, "%f %f %f\n", it.p2.x, it.p2.y, 0.);
		fprintf(fp, "%f %f %f\n", it.p3.x, it.p3.y, 0.);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELLS %d %d\n", num_tri, 3 * num_tri + num_tri);
	for (int i = 0; i < num_tri; i++) {
		fprintf(fp, "3 %d %d %d\n", 3 * i + 0, 3 * i + 1, 3 * i + 2);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELL_TYPES %d\n", num_tri);
	for (int i = 0; i < num_tri; i++) {
		fprintf(fp, "5\n");
	}

	fclose(fp);

}

void vtk_writer_write_FEM_tool(const tool_FEM* tool_FEM, int step)
{
	int num_node = tool_FEM->node_number;
	int num_tri = tool_FEM->element_number;


	float2_t* position_h = 0;
	float_t* T_h = 0;
	int4* nodes_in_element_h = 0;

	position_h = new float2_t[num_node];
	nodes_in_element_h = new int4[num_tri];
	T_h = new float_t[num_node];

	cudaMemcpy(position_h, tool_FEM->mesh_GPU->position, sizeof(float2_t) * (num_node), cudaMemcpyDeviceToHost);
	cudaMemcpy(nodes_in_element_h, tool_FEM->mesh_GPU->node_num, sizeof(int4) * (num_tri), cudaMemcpyDeviceToHost);
	cudaMemcpy(T_h, tool_FEM->FEM_solver->m_T, sizeof(float_t) * (num_node), cudaMemcpyDeviceToHost);

    char buf[256];
	sprintf(buf, "results/vtk_tool_%06d.vtk", step);
	FILE* fp = fopen(buf, "w+");

	fprintf(fp, "# vtk DataFile Version 2.0\n");
	fprintf(fp, "mfree iwf\n");
	fprintf(fp, "ASCII\n");
	fprintf(fp, "\n");
	fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
	fprintf(fp, "POINTS %d float\n", num_node);

	double x1, y1;
	for (int i = 0; i < num_node; i++) {
		x1 = position_h[i].x;
		y1 = position_h[i].y;
		fprintf(fp, "%f %f %f\n", x1, y1, 0.);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELLS %d %d\n", num_tri, 3 * num_tri + num_tri);
	for (int i = 0; i < num_tri; i++) {
		fprintf(fp, "3 %d %d %d\n", nodes_in_element_h[i].x, nodes_in_element_h[i].y, nodes_in_element_h[i].z);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELL_TYPES %d\n", num_tri);
	for (int i = 0; i < num_tri; i++) {
		fprintf(fp, "5\n");
	}

	fprintf(fp, "POINT_DATA %d\n", num_node);

	fprintf(fp, "SCALARS temperature float 1\n");		// Current particle density
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < num_node; i++) {
		fprintf(fp, "%f\n", T_h[i]);
	}
	fprintf(fp, "\n");

	fclose(fp);

	delete[] position_h;
	delete[] T_h;
	delete[] nodes_in_element_h;
}

void vtk_writer_write_FEM_segments(const tool_FEM* tool_FEM, int step){
#ifdef WEAR_NODE_SHIFT

	int seg_num = tool_FEM->cutting_segment_size;

	float2_t* left = 0;
	float2_t* right = 0;
	float2_t* n = 0;
	float4_t* wear = 0;
	float_t* sliding_force = 0;
	float_t* wear_rate = 0;


	left = new float2_t[seg_num];
	right = new float2_t[seg_num];
	n = new float2_t[seg_num];
	wear = new float4_t[seg_num];
	sliding_force = new float_t[seg_num];
	wear_rate = new float_t[seg_num];

	cudaMemcpy(left, tool_FEM->segments_FEM->left, sizeof(float2_t) * (seg_num), cudaMemcpyDeviceToHost);
	cudaMemcpy(right, tool_FEM->segments_FEM->right, sizeof(float2_t) * (seg_num), cudaMemcpyDeviceToHost);
	cudaMemcpy(n, tool_FEM->segments_FEM->n, sizeof(float2_t) * (seg_num), cudaMemcpyDeviceToHost);
	cudaMemcpy(wear, tool_FEM->segments_FEM->physical_para, sizeof(float4_t) * (seg_num), cudaMemcpyDeviceToHost);
	cudaMemcpy(sliding_force, tool_FEM->segments_FEM->sliding_force, sizeof(float_t) * (seg_num), cudaMemcpyDeviceToHost);
	cudaMemcpy(wear_rate, tool_FEM->segments_FEM->wear_nodes, sizeof(float_t) * (seg_num), cudaMemcpyDeviceToHost);

	char buf[256];
	sprintf(buf, "results/vtk_seg_%06d.vtk", step);
	FILE* fp = fopen(buf, "w+");

	fprintf(fp, "# vtk DataFile Version 2.0\n");
	fprintf(fp, "mfree iwf\n");
	fprintf(fp, "ASCII\n");
	fprintf(fp, "\n");
	fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
	fprintf(fp, "POINTS %d float\n", 3 * seg_num);

	double x1, y1;
	for (int i = 0; i < seg_num; i++) {
		x1 = left[i].x;
		y1 = left[i].y;
		fprintf(fp, "%f %f %f\n", x1, y1, 0.);
		x1 = right[i].x;
		y1 = right[i].y;
		fprintf(fp, "%f %f %f\n", x1, y1, 0.);
		float_t dis = sqrt((left[i].x - right[i].x) * (left[i].x - right[i].x) + (left[i].y - right[i].y) * (left[i].y - right[i].y));
		x1 = right[i].x + dis * n[i].x;
		y1 = right[i].y + dis * n[i].y;
		fprintf(fp, "%f %f %f\n", x1, y1, 0.);
	}
	//fprintf(fp, "%f %f %f\n", tl.x, tl.y, 0.);
	fprintf(fp, "\n");

	fprintf(fp, "CELLS %d %d\n", seg_num, 3 * seg_num + seg_num);
	for (int i = 0; i < seg_num; i++) {
		fprintf(fp, "3 %d %d %d \n", 3 * i, 3 * i + 1, 3 * i + 2);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELL_TYPES %d\n", seg_num);
	for (int i = 0; i < seg_num; i++) {
		fprintf(fp, "5\n");
	}

	fprintf(fp, "POINT_DATA %d\n", 3 * seg_num);

	fprintf(fp, "SCALARS PressureGPa float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < seg_num; i++) {
		fprintf(fp, "%f\n", 0.);
		fprintf(fp, "%f\n", 0.);
		if (wear[i].x != 0.) {
			fprintf(fp, "%f\n", wear[i].y / wear[i].x * 100.);

		}
		else {
			fprintf(fp, "%f\n", 0.);
		}
	}
	fprintf(fp, "\n");

	fprintf(fp, "SCALARS Velocity float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < seg_num; i++) {
		fprintf(fp, "%f\n", 0.);
		fprintf(fp, "%f\n", 0.);
		if (wear[i].x != 0.) {
			fprintf(fp, "%f\n", wear[i].z / wear[i].x * 10000.);
		}
		else {
			fprintf(fp, "%f\n", 0.);
		}
	}
	fprintf(fp, "\n");

	fprintf(fp, "SCALARS Temperature float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < seg_num; i++) {
		fprintf(fp, "%f\n", 0.);
		fprintf(fp, "%f\n", 0.);
		if (wear[i].x != 0.) {
			fprintf(fp, "%f\n", wear[i].w / wear[i].x);
		}
		else {
			fprintf(fp, "%f\n", 0.);
		}
	}
	fprintf(fp, "\n");

	fprintf(fp, "SCALARS Frictional_stress float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < seg_num; i++) {
		fprintf(fp, "%f\n", 0.);
		fprintf(fp, "%f\n", 0.);
		if (wear[i].x != 0.) {
			fprintf(fp, "%f\n", sliding_force[i] / wear[i].x * 100.);
		}
		else {
			fprintf(fp, "%f\n", 0.);
		}
	}
	fprintf(fp, "\n");

	fprintf(fp, "SCALARS Wear_rate float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < seg_num; i++) {
		fprintf(fp, "%f\n", 0.);
		fprintf(fp, "%f\n", 0.);
		if (wear[i].x != 0.) {
			fprintf(fp, "%f\n", wear_rate[i] * 1e10);
		}
		else {
			fprintf(fp, "%f\n", 0.);
		}
	}
	fprintf(fp, "\n");


	fclose(fp);

	delete[] left;
	delete[] right;
	delete[] n;
	delete[] wear;
	delete[] sliding_force;
	delete[] wear_rate;
#endif
}

void vtk_writer_write_tool_org(const tool_FEM* tool_FEM, int step, float2_t tl_ref)
{

	int num_node = tool_FEM->node_number;
	int num_tri = tool_FEM->element_number;

	vec2_t tl_current = tool_FEM->tl_point_ref;
	float2_t delta_x;
	delta_x.x = tl_current.x - tl_ref.x;
	delta_x.y = tl_current.y - tl_ref.y;


	float2_t* position_h = 0;
	float_t* T_h = 0;
	int4* nodes_in_element_h = 0;

	position_h = new float2_t[num_node];
	nodes_in_element_h = new int4[num_tri];
	T_h = new float_t[num_node];
	

	cudaMemcpy(position_h, tool_FEM->mesh_GPU->position, sizeof(float2_t) * (num_node), cudaMemcpyDeviceToHost);
	cudaMemcpy(nodes_in_element_h, tool_FEM->mesh_GPU->node_num, sizeof(int4) * (num_tri), cudaMemcpyDeviceToHost);
	cudaMemcpy(T_h, tool_FEM->FEM_solver->m_T, sizeof(float_t) * (num_node), cudaMemcpyDeviceToHost);

	char buf[256];
	sprintf(buf, "results/vtk_tool_org_%06d.vtk", step);
	FILE* fp = fopen(buf, "w+");

	fprintf(fp, "# vtk DataFile Version 2.0\n");
	fprintf(fp, "mfree iwf\n");
	fprintf(fp, "ASCII\n");
	fprintf(fp, "\n");
	fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
	fprintf(fp, "POINTS %d float\n", num_node);

	double x1, y1;
	for (int i = 0; i < num_node; i++) {
		x1 = position_h[i].x - delta_x.x;
		y1 = position_h[i].y - delta_x.y;
		fprintf(fp, "%f %f %f\n", x1, y1, 0.);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELLS %d %d\n", num_tri, 3 * num_tri + num_tri);
	for (int i = 0; i < num_tri; i++) {
		fprintf(fp, "3 %d %d %d\n", nodes_in_element_h[i].x, nodes_in_element_h[i].y, nodes_in_element_h[i].z);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELL_TYPES %d\n", num_tri);
	for (int i = 0; i < num_tri; i++) {
		fprintf(fp, "5\n");
	}

	fprintf(fp, "POINT_DATA %d\n", num_node);

	fprintf(fp, "SCALARS temperature float 1\n");		// Current particle density
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < num_node; i++) {
		fprintf(fp, "%f\n", T_h[i]);
	}
	fprintf(fp, "\n");

	fclose(fp);
	delete[] position_h;
	delete[] T_h;
	delete[] nodes_in_element_h;

}

void vtk_writer_write_contact_variables(const tool_FEM* tool_FEM, int step)
{
#ifdef WEAR_NODE_SHIFT
	int seg_num = tool_FEM->cutting_segment_size;

	static float2_t* left = 0;
	static float2_t* right = 0;
	static float2_t* n = 0;
	static float4_t* wear = 0;
	static float_t* sliding_force = 0;
	static float_t* wear_rate = 0;
	//static wear_constants  wear_parameters;



	if (step == 0) {

		left = new float2_t[seg_num];
		right = new float2_t[seg_num];
		n = new float2_t[seg_num];
		wear = new float4_t[seg_num];
		sliding_force = new float_t[seg_num];
		wear_rate = new float_t[seg_num];


	}

	cudaMemcpy(left, tool_FEM->segments_FEM->left, sizeof(float2_t) * (seg_num), cudaMemcpyDeviceToHost);
	cudaMemcpy(right, tool_FEM->segments_FEM->right, sizeof(float2_t) * (seg_num), cudaMemcpyDeviceToHost);
	cudaMemcpy(n, tool_FEM->segments_FEM->n, sizeof(float2_t) * (seg_num), cudaMemcpyDeviceToHost);

	cudaMemcpy(wear, tool_FEM->segments_FEM->physical_para, sizeof(float4_t) * (seg_num), cudaMemcpyDeviceToHost);
	cudaMemcpy(sliding_force, tool_FEM->segments_FEM->sliding_force, sizeof(float_t) * (seg_num), cudaMemcpyDeviceToHost);
	cudaMemcpy(wear_rate, tool_FEM->segments_FEM->wear_nodes, sizeof(float_t) * (seg_num), cudaMemcpyDeviceToHost);
	//cudaMemcpy(&wear_parameters, &wear_consts, sizeof(wear_constants) , cudaMemcpyDeviceToHost);

	//cudaMemcpyToSymbol(&wear_parameters, &wear_consts, sizeof(wear_constants), 0, cudaMemcpyDeviceToHost);

	char buf[256];
	sprintf(buf, "results/vtk_phy_wear_%06d.vtk", step);
	FILE* fp = fopen(buf, "w+");

	fprintf(fp, "# vtk DataFile Version 2.0\n");
	fprintf(fp, "mfree iwf\n");
	fprintf(fp, "ASCII\n");
	fprintf(fp, "\n");


	fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
	fprintf(fp, "POINTS %d float\n", seg_num);
	for (unsigned int i = 0; i < seg_num; i++) {
		fprintf(fp, "%f %f %f\n", 0.5 * (left[i].x + right[i].x), 0.5 * (left[i].y + right[i].y), 0.);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELLS %d %d\n", seg_num, 2 * seg_num);
	for (unsigned int i = 0; i < seg_num; i++) {
		fprintf(fp, "%d %d\n", 1, i);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELL_TYPES %d\n", seg_num);
	for (unsigned int i = 0; i < seg_num; i++) {
		fprintf(fp, "%d\n", 1);
	}
	fprintf(fp, "\n");

	fprintf(fp, "POINT_DATA %d\n", seg_num);

	fprintf(fp, "SCALARS PressureGPa float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < seg_num; i++) {
		if (wear[i].x != 0.) {
			fprintf(fp, "%f\n", 100. * wear[i].y / wear[i].x );

		}
		else {
			fprintf(fp, "%f\n", 0.);
		}
	}
	fprintf(fp, "\n");



	fprintf(fp, "SCALARS Velocity float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < seg_num; i++) {
		if (wear[i].x != 0.) {
			fprintf(fp, "%f\n", wear[i].z / wear[i].x * 10000.);
		}
		else {
			fprintf(fp, "%f\n", 0.);
		}
	}
	fprintf(fp, "\n");

	fprintf(fp, "SCALARS Temperature float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < seg_num; i++) {
		if (wear[i].x != 0.) {
			fprintf(fp, "%f\n", wear[i].w / wear[i].x);
		}
		else {
			fprintf(fp, "%f\n", 0.);
		}
	}
	fprintf(fp, "\n");

	fprintf(fp, "SCALARS Frictional_stress float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < seg_num; i++) {
		if (wear[i].x != 0.) {
			fprintf(fp, "%f\n", sliding_force[i] / wear[i].x * 100.);
		}
		else {
			fprintf(fp, "%f\n", 0.);
		}
	}
	fprintf(fp, "\n");

	fprintf(fp, "SCALARS Wear_rate float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < seg_num; i++) {
		fprintf(fp, "%f\n", 0.);
		fprintf(fp, "%f\n", 0.);
		if (wear[i].x != 0.) {
			fprintf(fp, "%f\n", wear_rate[i] * 1e10);
		}
		else {
			fprintf(fp, "%f\n", 0.);
		}
	}
	fprintf(fp, "\n");

	fclose(fp);
#endif
}

