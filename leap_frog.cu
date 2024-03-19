// Copyright ETH Zurich, IWF
// SPH-FEM cutting simulations on the basis of mfree_iwf-ul_cut_gpu: 
// https://github.com/iwf-inspire/mfree_iwf-ul_cut_gpu
// Date: 2024.03.10

// You should have received a copy of the GNU General Public License
// along with mfree_iwf-ul_cut_gpu.  If not, see <http://www.gnu.org/licenses/>.


#include "leap_frog.h"

#include "grid_gpu_green.h"

extern bool wear_pro;

struct inistate_struct {
	float2_t *pos_init;
	float2_t *vel_init;
	float4_t *S_init;
	float_t  *rho_init;
	float_t  *T_init;
	float_t  *T_init_tool;
};

__global__ void init(particle_gpu particles, inistate_struct inistate) {

	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N) return;
	if (particles.blanked[pidx] == 1.) return;

	inistate.pos_init[pidx] = particles.pos[pidx];
	inistate.vel_init[pidx] = particles.vel[pidx];
	inistate.S_init[pidx]   = particles.S[pidx];
	inistate.rho_init[pidx] = particles.rho[pidx];
	inistate.T_init[pidx]   = particles.T[pidx];

}

__global__ void init_FEM_tool(tool_FEM global_tool_FEM, inistate_struct inistate) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >=global_tool_FEM.node_number) return;
	inistate.T_init_tool[pidx] = global_tool_FEM.mesh_GPU->T_nodes[pidx];
}

__global__ void predict(particle_gpu particles, inistate_struct inistate, float_t dt) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N) return;
	if (particles.blanked[pidx] == 1.) return;

	particles.pos[pidx].x = inistate.pos_init[pidx].x + 0.5*dt*particles.pos_t[pidx].x;
	particles.pos[pidx].y = inistate.pos_init[pidx].y + 0.5*dt*particles.pos_t[pidx].y;

	particles.vel[pidx].x = inistate.vel_init[pidx].x + 0.5*dt*particles.vel_t[pidx].x;
	particles.vel[pidx].y = inistate.vel_init[pidx].y + 0.5*dt*particles.vel_t[pidx].y;

#ifdef TVF
	particles.vel_adv[pidx].x = inistate.vel_init[pidx].x + 0.5*dt*particles.vel_adv_t[pidx].x;
	particles.vel_adv[pidx].y = inistate.vel_init[pidx].y + 0.5*dt*particles.vel_adv_t[pidx].y;
#endif

	particles.S[pidx].x   = inistate.S_init[pidx].x + 0.5*dt*particles.S_t[pidx].x;
	particles.S[pidx].y   = inistate.S_init[pidx].y + 0.5*dt*particles.S_t[pidx].y;
	particles.S[pidx].z   = inistate.S_init[pidx].z + 0.5*dt*particles.S_t[pidx].z;
	particles.S[pidx].w   = inistate.S_init[pidx].w + 0.5*dt*particles.S_t[pidx].w;

	particles.rho[pidx]   = inistate.rho_init[pidx] + 0.5*dt*particles.rho_t[pidx];

	particles.T[pidx]     = inistate.T_init[pidx]   + 0.5*dt*particles.T_t[pidx];
}

__global__ void correct(particle_gpu particles, inistate_struct inistate, float_t dt) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N) return;
	if (particles.blanked[pidx] == 1.) return;

	particles.pos[pidx].x = inistate.pos_init[pidx].x + dt*particles.pos_t[pidx].x;
	particles.pos[pidx].y = inistate.pos_init[pidx].y + dt*particles.pos_t[pidx].y;

	particles.vel[pidx].x = inistate.vel_init[pidx].x + dt*particles.vel_t[pidx].x;
	particles.vel[pidx].y = inistate.vel_init[pidx].y + dt*particles.vel_t[pidx].y;

#ifdef TVF
	particles.vel_adv[pidx].x = inistate.vel_init[pidx].x + dt*particles.vel_adv_t[pidx].x;
	particles.vel_adv[pidx].y = inistate.vel_init[pidx].y + dt*particles.vel_adv_t[pidx].y;
#endif

	particles.S[pidx].x   = inistate.S_init[pidx].x + dt*particles.S_t[pidx].x;
	particles.S[pidx].y   = inistate.S_init[pidx].y + dt*particles.S_t[pidx].y;
	particles.S[pidx].z   = inistate.S_init[pidx].z + dt*particles.S_t[pidx].z;
	particles.S[pidx].w   = inistate.S_init[pidx].w + dt*particles.S_t[pidx].w;

	particles.rho[pidx]   = inistate.rho_init[pidx] + dt*particles.rho_t[pidx];

	particles.T[pidx]     = inistate.T_init[pidx]   + dt*particles.T_t[pidx];

}


void leap_frog::step(particle_gpu *particles, grid_base *g) {
	dim3 dG((particles->N + BLOCK_SIZE-1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);

	inistate_struct inistate;
	inistate.pos_init = pos_init;
	inistate.vel_init = vel_init;
	inistate.S_init   = S_init;
	inistate.rho_init = rho_init;
	inistate.T_init   = T_init;

	//spatial sorting
	g->update_geometry(particles, global_tool, 2.);
	g->assign_hashes(particles, global_tool);
	g->sort(particles, global_tool);
	g->get_cells(particles, cell_start, cell_end);

	//leap frog predictor
	init<<<dG,dB>>>(*particles, inistate);
	predict<<<dG,dB>>>(*particles, inistate, global_dt);

	material_eos(particles);

	corrector_artificial_stress(particles);

	interactions_setup_geometry_constants(g);
	interactions_monaghan(particles, cell_start, cell_end, g->num_cell());

#ifdef Thermal_Conduction_PSE
	interactions_heat_pse(particles, cell_start, cell_end, g->num_cell());
#endif

	material_stress_rate_jaumann(particles);

	contmech_continuity(particles);
	contmech_momentum(particles);
	contmech_advection(particles);

	//leap frog predictor
	correct<<<dG,dB>>>(*particles, inistate, global_dt);

	if (global_tool) {
		material_fric_heat_gen(particles, global_tool->get_vel());
	}

	//plastic state by radial return
	plasticity_johnson_cook(particles);

	//establish contact by penalty method if tool is present
	if (global_tool) {
		tool_gpu_update_tool(global_tool, particles);
		compute_contact_forces(particles);
	}

	//boundary conditions
	perform_boundary_conditions(particles);
	perform_boundary_conditions_thermal(particles);

	//debugging methods
//	debug_check_valid(particles);
	debug_invalidate(particles);
}

void leap_frog::step(particle_gpu* particles, grid_base* g, tool_FEM* m_tool)
{
	dim3 dG((particles->N + BLOCK_SIZE - 1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);

	inistate_struct inistate;
	inistate.pos_init = pos_init;
	inistate.vel_init = vel_init;
	inistate.S_init = S_init;
	inistate.rho_init = rho_init;
	inistate.T_init = T_init;
	//if (wear_pro) {printf("Point 0\n");}
	//spatial sorting
	g->update_geometry(particles, global_tool, 2.);
	g->assign_hashes(particles, global_tool);
	g->sort(particles, global_tool);
	g->get_cells(particles, cell_start, cell_end);
	//if (wear_pro) {printf("Point 1\n");}

	init << <dG, dB >> > (*particles, inistate);
	m_tool->FEM_solver->reset_FEM_server();

#ifdef WEAR_NODE_SHIFT
		m_tool->segments_wear_rate_reset();
#endif
    //if (wear_pro) {printf("Point 2\n");}
	predict << <dG, dB >> > (*particles, inistate, global_dt);
	m_tool->FEM_solver->calculate_temp(0.5 * global_dt);    

	material_eos(particles);

	corrector_artificial_stress(particles);

	interactions_setup_geometry_constants(g);
	interactions_monaghan(particles, cell_start, cell_end, g->num_cell());

#ifdef Thermal_Conduction_PSE
	interactions_heat_pse(particles, cell_start, cell_end, g->num_cell());
#endif

	material_stress_rate_jaumann(particles);

	contmech_continuity(particles);
	contmech_momentum(particles);
	contmech_advection(particles);


	//leap frog corrector
	correct << <dG, dB >> > (*particles, inistate, global_dt);

	// Change of boundary condition in FEM solver
	//material_fric_heat_gen_FEM_v1(particles, m_tool->segments_FEM, global_tool_FEM->get_vel());
	material_fric_heat_gen_FEM_v2(particles, m_tool->segments_FEM, m_tool->get_vel(), m_tool);
	// FEM solver
	m_tool->apply_heat_boundary_condition();
	m_tool->FEM_solver->calculate_temp_rate();        // The correction can be done in a class function

	m_tool->FEM_solver->calculate_temp(global_dt);

	//plastic state by radial return
	plasticity_johnson_cook(particles);
	
#ifdef WEAR_NODE_SHIFT
	if (wear_pro) {
		// tool nodes adjustment
		wear_adjustment(m_tool);
		//m_tool->segments_wear_rate_reset();
	}
#endif
	
	//establish contact by penalty method if tool is present
	// first move the tool
	m_tool->segments_update(global_dt);
	m_tool->tool_update_FEM_gpu(global_dt);

		
	//compute_contact_forces_FEM_tool(particles, m_tool);
	compute_contact_forces_FEM_tool_v3(particles, m_tool->segments_FEM, m_tool);

	//boundary conditions
	perform_boundary_conditions(particles);
	perform_boundary_conditions_thermal(particles);
	
	//debugging methods
	//debug_check_valid(particles);
	debug_invalidate(particles);
}

leap_frog::leap_frog(unsigned int num_part, unsigned int num_cell) {
	cudaMalloc((void **) &pos_init, sizeof(float2_t)*num_part);
	cudaMalloc((void **) &vel_init, sizeof(float2_t)*num_part);
	cudaMalloc((void **) &S_init,   sizeof(float4_t)*num_part);
	cudaMalloc((void **) &rho_init,   sizeof(float_t)*num_part);
	cudaMalloc((void **) &T_init,   sizeof(float_t)*num_part);

	cudaMalloc((void **) &cell_start, sizeof(int)*num_cell);
	cudaMalloc((void **) &cell_end,   sizeof(int)*num_cell);
}

leap_frog::leap_frog(unsigned int num_part, unsigned int num_cell, int node_num_in_FEM_tool)
{
	cudaMalloc((void**)&pos_init, sizeof(float2_t) * num_part);
	cudaMalloc((void**)&vel_init, sizeof(float2_t) * num_part);
	cudaMalloc((void**)&S_init, sizeof(float4_t) * num_part);
	cudaMalloc((void**)&rho_init, sizeof(float_t) * num_part);
	cudaMalloc((void**)&T_init, sizeof(float_t) * num_part);

	cudaMalloc((void**)&cell_start, sizeof(int) * num_cell);
	cudaMalloc((void**)&cell_end, sizeof(int) * num_cell);
}
