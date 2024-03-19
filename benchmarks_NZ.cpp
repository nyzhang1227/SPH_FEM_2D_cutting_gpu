// Copyright ETH Zurich, IWF
// SPH-FEM cutting simulations on the basis of mfree_iwf-ul_cut_gpu: 
// https://github.com/iwf-inspire/mfree_iwf-ul_cut_gpu
// Date: 2024.03.10

// You should have received a copy of the GNU General Public License
// along with mfree_iwf-ul_cut_gpu.  If not, see <http://www.gnu.org/licenses/>.

#include "benchmarks_NZ.h"

particle_gpu* setup_ref_cut_FEM_tool(int ny, grid_base** grid, float_t rake, float_t clear, float_t chamfer, float_t speed, float_t feed) {

}

particle_gpu* setup_ref_cut_FEM_tool_no_wear(int ny, grid_base** grid, float_t rake, float_t clear, float_t chamfer, float_t speed, float_t feed) {
	
}

particle_gpu* setup_ref_cut_FEM_tool_gmsh_Ti64(int ny, grid_base** grid, float_t rake, float_t clear, float_t chamfer, float_t speed, float_t feed) {
	
}

particle_gpu* setup_ref_cut_FEM_tool_gmsh_Ck45(int ny, grid_base** grid, float_t rake, float_t clear, float_t chamfer, float_t speed, float_t feed) {
	
}

particle_gpu* setup_ref_cut_FEM_tool_gmsh_friction_paper_try(int ny, grid_base** grid, float_t rake, float_t clear, float_t chamfer, float_t speed, float_t feed) {
	
}

particle_gpu* setup_ref_cut_FEM_tool_gmsh_friction_paper_try_large_feed(int ny, grid_base** grid, float_t rake, float_t clear, float_t chamfer, float_t speed, float_t feed) {
	
}

particle_gpu* setup_ref_cut_FEM_tool_gmsh_wear_test_large_feed_Ti64(int ny, grid_base** grid, float_t rake, float_t clear, float_t chamfer, float_t speed, float_t feed) {
	
}

particle_gpu* setup_ref_cut_FEM_tool_gmsh_wear_test(int ny, grid_base** grid, float_t rake, float_t clear, float_t chamfer, float_t speed, float_t feed) {
	
}

particle_gpu* setup_ref_cut_FEM_tool_gmsh_wear_test_Ck45(int ny, grid_base** grid, float_t rake, float_t clear, float_t chamfer, float_t speed, float_t feed) {
	
}

particle_gpu* setup_ref_cut_FEM_tool_gmsh_wear_test_Ck45_flank(int ny, grid_base** grid, float_t rake, float_t clear, float_t chamfer, float_t speed, float_t feed) {
	
}

particle_gpu* setup_ref_cut_FEM_tool_no_wear_thermal_paper(int ny, grid_base** grid, float_t rake, float_t clear, float_t chamfer, float_t speed, float_t feed) {
	phys_constants phys = make_phys_constants();
	corr_constants corr = make_corr_constants();
	joco_constants joco = make_joco_constants();
	trml_constants trml_wp = make_trml_constants();
	trml_constants trml_tool = make_trml_constants();
	tool_constants tool_const = make_tool_constants();
	wear_constants wear_con = make_wear_constants();

	bool use_thermals = true;		//set thermal constants?
	bool sample_tool = true;		//expand thermal solver to tool?
    
    feed = 0.01;
    speed = 83.333328*1e-5 / 500 * 318;
	ny = 210;
	//dimensions of work piece
	float_t hi_x = 0.130; float_t hi_y = 0.060;
	float_t lo_x = 0.005; float_t lo_y = hi_y - 3 * feed;

	

	float_t dy = (hi_y - lo_y) / (ny - 1);
	float_t dx = dy;
	int nx = (hi_x - lo_x) / dx;
	float_t cutting_distance = 0.06;	//cut 1 mm of material
	
	//h = hdx*dx
	float_t hdx = 1.5;

	//Ti6Al4v according to Johnson 85
	phys.E = 1.1; // 1.138
	phys.nu = 0.35; // 0.35
	phys.rho0 = 4.43;
	phys.G = phys.E / (2. * (1. + phys.nu));
	phys.K = 2.0 * phys.G * (1 + phys.nu) / (3 * (1 - 2 * phys.nu));
	phys.mass = dx * dx * phys.rho0;
	
	
	
	joco.A = 0.0085200;
	joco.B = 0.0033890;
	joco.C = 0.02754;
	joco.m = 0.5961;
	joco.n = 0.148;
	joco.Tref = 300.;
	joco.Tmelt = 1836.0000;
	joco.eps_dot_ref = 1e-6;


	joco.tanh_a = 5;
	joco.tanh_b = 0.6;
	joco.tanh_c = 600.;
	joco.tanh_d = 0.7;
    
   
	float_t rho_tool = 15.25;   //15.25
	phys.mass_tool = dx * dx * rho_tool;
	if (use_thermals) {
		// https://www.azom.com/properties.aspx?ArticleID=1203
		trml_wp.cp = 526 * 1e-8;			// Heat Capacity 580 526
		trml_wp.tq = 0.9;				// Taylor-Quinney Coefficient
		trml_wp.k = 6.8 * 1e-13;			// Thermal Conduction 7.3 6.8
		trml_wp.alpha = trml_wp.k / (phys.rho0 * trml_wp.cp);	// Thermal diffusivity
		trml_wp.eta = 0.9;
		trml_wp.T_init = joco.Tref;

		//www.azom.com/properties.aspx?ArticleID=1203
		trml_tool.cp = 15 * 1e-08;
		trml_tool.tq = .0;
		trml_tool.k = 59 * 1e-13;
		trml_tool.alpha = trml_tool.k / (rho_tool * trml_tool.cp);
		trml_tool.T_init = joco.Tref;
	}

	//artificial viscosity, XSPH and artificial stress constants
	corr.alpha = 1.;
	corr.beta = 1.;
	corr.eta = 0.1;
	corr.xspheps = 0.5;
	corr.stresseps = 0.3;
	{
		float_t h1 = 1. / (hdx * dx);
		float_t q = dx * h1;
		float_t fac = 10 * (M_1_PI) / 7.0 * h1 * h1;
		corr.wdeltap = fac * (1 - 1.5 * q * q * (1 - 0.5 * q));
	}

	//generate particles in work piece
	int n = nx * ny;
	float2_t* pos = new float2_t[n];
	int part_iter = 0;

	float_t min_x = FLT_MAX;
	float_t max_x = -FLT_MAX;

	float_t min_y = FLT_MAX;
	float_t max_y = -FLT_MAX;

	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++) {
			pos[part_iter].x = i * dx + lo_x;
			pos[part_iter].y = j * dx + lo_y;

			max_x = fmax(max_x, pos[part_iter].x);
			min_x = fmin(min_x, pos[part_iter].x);
			max_y = fmax(max_y, pos[part_iter].y);
			min_y = fmin(min_y, pos[part_iter].y);

			part_iter++;
		}
	}

	//remaining initial and boundary conditions
	float2_t* vel = new float2_t[n];
	float_t* rho = new float_t[n];
	float_t* h = new float_t[n];
	float_t* T = new float_t[n];
	float_t* fixed = new float_t[n];
	float_t* tool_p = new float_t[n];

	for (int i = 0; i < n; i++) {
		rho[i] = phys.rho0;
		h[i] = hdx * dx;
		vel[i].x = 0.;
		vel[i].y = 0.;
		T[i] = joco.Tref;
		//fix bottom
		fixed[i] = (pos[i].y < min_y + 0.5 * dx) ? 1. : 0.;
		//fix left hand side below blade
		fixed[i] = fixed[i] || (pos[i].x < min_x + dx / 2 && pos[i].y < min_y + 0.5 * (hi_y - lo_y));
		//fix all of right hand side
		fixed[i] = fixed[i] || (pos[i].x > max_x - dx);

		tool_p[i] = 0.;
	}

	//tool velocity and friction constant
	float_t vel_tool = (speed == 0.f) ? 83.333328 * 1e-5 : speed; //500m/min
	
    float_t mu_fric = 0.65;
	//set final tool constants	   
	float_t contact_alpha = 0.02;	// NOTE, IMPORTANT: needs to be reduced if very small timesteps are necessary, my simulation: 0.08
	// set up the tool constants
	tool_constants tc_h = make_tool_constants();
	tc_h.cp = 292 * 1e-8;  // 292
	tc_h.tq = 0.9;
	tc_h.k = 88 * 1e-13;  // 88
	tc_h.eta = 1.0;
	tc_h.rho = 15.25;    // 15.25
	tc_h.T0 = joco.Tref;
	tc_h.h = 2e-10;           // 1e-15 for unit
	tc_h.beta = 0.5;
	tc_h.mu = mu_fric;
	tc_h.contact_alpha = contact_alpha;
	tc_h.slave_mass = phys.mass;
#ifdef SHEAR_LIMIT_FRI
	tc_h.shear_limit = 1.0;
#endif // SHEAR_LIMIT_FRI
#ifdef VELOCITY_FRI
	tc_h.mu = 0.627;
	tc_h.exp_co_vel = 0.154;
#endif // VELOCITY_FRI
#ifdef TEMP_FRI
	tc_h.mu = 0.51;
	tc_h.exp_co_temp = 5.76;
	tc_h.T_melt = joco.Tmelt;
#endif // TEMP_FRI

	//-----------------------------------------

	//tool dimensions copied from ruttimanns FEM simualations
	float_t nudge = 0.004;
	glm::dvec2 tl(-0.05 + nudge, 0.0986074);

	float_t l_tool = 2.0 * ( -0.0086824 - -0.08);
	float_t h_tool = 2.0 * ( 0.1286074 - 0.0555074);

	tool_FEM* t = new tool_FEM(tl, l_tool, h_tool, rake, clear, chamfer, mu_fric);
	global_tool_FEM = t;
	

	// generate the mesh in CPU

	t->mesh_CPU = new mesh_read("Mesh_veryfine_r5_rake0_clearance11.inp");
	

#ifdef USE_HOT_TOOL
    t->mesh_CPU->mesh_read_temperature("results/vtk_tool_000100.vtk");
#endif

	// initiate the mesh in GPU
	t->mesh_GPU = new mesh_GPU_m(t->mesh_CPU->nodes, t->mesh_CPU->elements, t->mesh_CPU->color_total, t->mesh_CPU->color_cutting_edge, 1);

	t->set_up_segments_from_mesh();
	t->set_up_FEM_tool_gpu(tc_h);

	//move tool to target feed
	
		float_t distance = t->right_point  - lo_x + dx*0.01;
		float_t time = distance / vel_tool;
		t->set_vel(glm::dvec2(vel_tool, 0.));
		t->update_tool(-time);
		t->tool_update_FEM_gpu(-time);
		t->segments_update(-time);
		

		float_t target_feed = feed;
		float_t current_feed = hi_y - t->lowest_point;
		float_t dist_to_target_feed = fabs(current_feed - target_feed);
		float_t correction_time = dist_to_target_feed / vel_tool;
		float_t sign = (current_feed > target_feed) ? 1 : -1.;
		t->set_vel(glm::dvec2(0., vel_tool));
		t->update_tool(correction_time * sign);
		t->tool_update_FEM_gpu(correction_time * sign);
		t->segments_update(correction_time * sign);
	
	t->set_vel(glm::dvec2(vel_tool, 0.));

	int n_tool = 0;
	
	//measure Bounding Box of complete domain
	float2_t bbmin = make_float2_t(FLT_MAX, FLT_MAX);
	float2_t bbmax = make_float2_t(-FLT_MAX, -FLT_MAX);
	for (unsigned int i = 0; i < n + n_tool; i++) {
		bbmin.x = fmin(pos[i].x, bbmin.x);
		bbmin.y = fmin(pos[i].y, bbmin.y);
		bbmax.x = fmax(pos[i].x, bbmax.x);
		bbmax.y = fmax(pos[i].y, bbmax.y);
	}

	bbmin.x -= 1e-8;
	bbmin.y -= 1e-8;

	bbmax.x += 1e-8 + 300 * dx;
	bbmax.y += 1e-8 + 500 * dx;

	float_t max_height = hi_y + hi_x - lo_x;
	bbmax.y = fmax(bbmax.y, max_height);

	//set up grid for spatial hashing
	//	- grid rothlin saves memory but is slower than grid green
	//	- each grid can be configured to adapt to the solution domain or to stay fixed
	//	- fixed grids work considerably faster than adapting ones

	//	*grid = new grid_gpu_green(10*n, n);
	*grid = new grid_gpu_green(n + n_tool, bbmin, bbmax, hdx * dx);
	//	*grid = new grid_gpu_rothlin(10*n, n);
	//	*grid = new grid_gpu_rothlin(n, bbmin, bbmax, hdx*dx);

	//check whether grid was set up correctly
	for (unsigned int i = 0; i < n + n_tool; i++) {
		bool in_x = pos[i].x > bbmin.x && pos[i].x < bbmax.x;
		bool in_y = pos[i].y > bbmin.y && pos[i].y < bbmax.y;
		if (!(in_x && in_y)) {
			printf("WARINING: particle out of inital bounding box!\n");
		}
	}

	//usui wear model
	float_t usui_K = 0.86; 	//1/GPa, 7.8
	usui_K = 1. * 100 * 0.86;	//bomb units
	float_t usui_alpha = 18769.9; //2500
	global_wear = new tool_wear(usui_K, usui_alpha, (unsigned int)n + n_tool, phys, glm::dvec2(0., vel_tool));

#ifdef WEAR_NODE_SHIFT

  #ifdef USE_ABRASIVE_WEAR
    wear_con.A = 2.37e-11 * 1000000000/2.;
	wear_con.B = 0.;
	wear_con.C = 0.;
	wear_con.D = 0.;
  #endif


#ifdef  USE_DIFFUSIVE_WEAR   // List et al. for C
    wear_con.A = 0.;
	wear_con.B = 0.;
	wear_con.C = 2.36e-8 * 50000;
	wear_con.D = 7860;
#endif
#ifdef USE_ZANGER_WEAR   
	// Combined wear model, Zanger Schulze
	wear_con.A = 0.387 *100000;
	wear_con.B = 17465.2 / 1.0 ;
	wear_con.C = 0.28 * 1e-5 *100000;
	wear_con.D = 18769.9 / 1.0 ;
#endif
#endif

	//propagate constants to actions and interactions
	actions_setup_corrector_constants(corr);
	actions_setup_physical_constants(phys);
	actions_setup_johnson_cook_constants(joco);
	actions_setup_thermal_constants_wp(trml_wp);
	actions_setup_thermal_constants_tool(trml_tool);
	actions_setup_tool_constants(tc_h);
#ifdef WEAR_NODE_SHIFT
	actions_setup_wear_constants(wear_con);
	t->FEM_tool_setup_wear_constants(wear_con);
#endif
	interactions_setup_geometry_constants(*grid);
	interactions_setup_corrector_constants(corr);
	interactions_setup_physical_constants(phys);
	//interactions_setup_thermal_constants_tool(trml_tool, global_tool);
	interactions_setup_thermal_constants_workpiece(trml_wp);
	interactions_setup_tool_constants(tc_h);


	//CFL based choice of time step
	global_dt = 0.1 * hdx * dx / (sqrt(phys.K / phys.rho0) + sqrt(vel_tool));
	global_t_final = cutting_distance / vel_tool;
	printf("timestep chosen: %e\n", global_dt);
	printf("calculating with %d regular and %d tool particles for a total of %d\n", n, n_tool, n + n_tool);

	//return new particle_gpu(pos, vel, rho, T, h, fixed,tool_p, n + n_tool);
	return new particle_gpu(pos, vel, rho, T, h, fixed, n );
}

particle_gpu* setup_ref_cut_FEM_tool_textured(int ny, grid_base** grid, float_t rake, float_t clear, float_t chamfer, float_t speed, float_t feed) {
	
}