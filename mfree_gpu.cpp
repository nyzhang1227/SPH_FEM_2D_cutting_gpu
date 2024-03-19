// Copyright ETH Zurich, IWF
// SPH-FEM cutting simulations on the basis of mfree_iwf-ul_cut_gpu: 
// https://github.com/iwf-inspire/mfree_iwf-ul_cut_gpu
// Date: 2024.03.10

// You should have received a copy of the GNU General Public License
// along with mfree_iwf-ul_cut_gpu.  If not, see <http://www.gnu.org/licenses/>.


#define _USE_MATH_DEFINES
#include <cmath>

#include <cuda_profiler_api.h>
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <nvml.h>
#include <sys/stat.h>
#include <sys/types.h>
//#include <direct.h>
#include <nvml.h>

#include "benchmarks.h"
#include "benchmarks_NZ.h"
#include "debug.h"
#include "grid_gpu_rothlin.h"
#include "leap_frog.h"
#include "output.h"
#include "types.h"
#include "tool.h"
#include "tool_wear.h"
#include "vtk_writer.h"


int global_step = 0;
tool *global_tool = 0;
tool_FEM *global_tool_FEM = 0;
tool_wear *global_wear = 0;
bool global_analyze_heat = true;

extern float_t global_dt;
extern float_t global_t_final;
bool wear_pro = false;

static bool log_wear = true;
/*
int poll_temp() {
	FILE *in;
	char buff[512];

	if(!(in = popen("nvidia-smi | grep '[0-9][0-9]C' | awk '{print $3}' | sed 's/C//'", "r"))){
		return -1;
	}

	if (fgets(buff, sizeof(buff), in)!=NULL){
		int temp;
		sscanf(buff, "%d", &temp);
		pclose(in);

		if (temp >= 82) {
			exit(0);
		}

		return temp;
	}

	pclose(in);
	return -1;
}
*/
void poll_memory_usage() {
        size_t free_byte ;
        size_t total_byte ;

        cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;

        if ( cudaSuccess != cuda_status ){
            printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
            exit(1);
        }

        double free_db = (double)free_byte ;
        double total_db = (double)total_byte ;
        double used_db = total_db - free_db ;
        printf("GPU memory usage: used = %f MB, free = %f MB, total = %f MB\n",
            used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}

void chose_cuda_device() {
	//init nvml
	nvmlInit();

	//list devices
	unsigned int device_count = 0;
	nvmlDeviceGetCount(&device_count);
	printf("found %d cuda devices!\n", device_count);

	//try to find a free device
	int free_device = -1;
	for (unsigned int i = 0; i < device_count; i++) {
        nvmlDevice_t device;
        nvmlDeviceGetHandleByIndex(i, &device);

		unsigned int info_count;
		nvmlProcessInfo_t infos;
		auto ret = nvmlDeviceGetComputeRunningProcesses (device , &info_count, &infos);

		if (ret == NVML_ERROR_NOT_SUPPORTED) {
			printf("nvidia believes that only professional cards may report if they are busy. simply goanna chose device 0\n");
			free_device = 0;
			break;
		}

		if (info_count > 0) {
			printf("device %d is busy with pid: %d\n", i, infos.pid);
		} else {
			printf("found free device %d\n", i);
			free_device = i;
			break;
		}
	}

	if (free_device == -1) {
		printf("no free cuda devices!\n");
		exit(-1);
	}

	cudaError_t err = cudaSetDevice(free_device);
	if (err != cudaError_t::cudaSuccess) {
		printf("couldn't set device!\n");
		exit(-1);
	} else {
		printf("context established!\n");
	}
}

int main(int argc, char *argv[]) {
	//cudaSetDevice(0);
	chose_cuda_device();
/*
#if defined Thermal_Conduction_Brookshaw and defined Thermal_Conduction_PSE
	printf("chose only one method for the thermal solver\n");
#endif
*/
#ifdef USE_DOUBLE
	printf("solver is running with double precision\n");
#else
	printf("solver is running with single precision\n");
#endif

	//default cut, sim according to ruttimann 2012
	double rake_angle = 0.;
	double clearance_angle = 4.5;
	double fillet = 0.00035;
	
	int ny =30; //same effective resolution as LSDYNA sims in ruttimann 2012
	double feed = 0.01;
	double speed_in_ui = 300;

	//opterr = 0;
	int c;

	printf("number of arguments %d\n", argc);
	for (int i = 0; i < argc; i++) {
		printf("%s\n", argv[i]);
	}

	//scan for arguments
	
	if (argc != 0) {
		while ((c = getopt (argc, argv, "r:c:f:v:n:q:")) != -1) {
			switch (c)
			{
			case 'r':
				sscanf(optarg, "%lf", &rake_angle);
				break;
			case 'c':
				sscanf(optarg, "%lf", &clearance_angle);
				break;
			case 'f':
				sscanf(optarg, "%lf", &fillet);
				break;
			case 'v':
				sscanf(optarg, "%lf", &speed_in_ui);
				break;
			case 'n':
				sscanf(optarg, "%d", &ny);
				break;
			case 'q':
				sscanf(optarg, "%lf", &feed);
				break;
			}
		}
	}
    double speed = 83.333328*1e-5 / 200 * speed_in_ui; //70m/min (254, 318)
	printf("arguments received %.9g %.9g %.9g %.9g %.9g\n", (float_t) rake_angle, (float_t) clearance_angle, (float_t) fillet, (float_t) speed_in_ui, (float_t) feed);
    
	grid_base *grid;
	//cutting simulation
#ifdef USE_FEM_TOOL
	//particle_gpu* particles = setup_ref_cut_FEM_tool(ny, &grid, rake_angle, clearance_angle, fillet, speed, feed);
	//particle_gpu* particles = setup_ref_cut_FEM_tool_no_wear(ny, &grid, rake_angle, clearance_angle, fillet, speed, feed);
	//particle_gpu* particles = setup_ref_cut_FEM_tool_gmsh_friction_paper_try(ny, &grid, rake_angle, clearance_angle, fillet, speed, feed);
	//particle_gpu* particles = setup_ref_cut_FEM_tool_gmsh_friction_paper_try_large_feed(ny, &grid, rake_angle, clearance_angle, fillet, speed, feed);
	//particle_gpu* particles = setup_ref_cut_FEM_tool_gmsh_wear_test_large_feed_Ti64(ny, &grid, rake_angle, clearance_angle, fillet, speed, feed);
	particle_gpu* particles = setup_ref_cut_FEM_tool_gmsh_wear_test(ny, &grid, rake_angle, clearance_angle, fillet, speed, feed);
	//particle_gpu* particles = setup_ref_cut_FEM_tool_gmsh_wear_test_Ck45(ny, &grid, rake_angle, clearance_angle, fillet, speed, feed);
	//particle_gpu* particles = setup_ref_cut_FEM_tool_gmsh_wear_test_Ck45_flank(ny, &grid, rake_angle, clearance_angle, fillet, speed, feed);
	//particle_gpu* particles = setup_ref_cut_FEM_tool_gmsh_Ti64(ny, &grid, rake_angle, clearance_angle, fillet, speed, feed);
	//particle_gpu* particles = setup_ref_cut_FEM_tool_gmsh_Ck45(ny, &grid, rake_angle, clearance_angle, fillet, speed, feed);
	//particle_gpu* particles = setup_ref_cut_FEM_tool_no_wear_thermal_paper(ny, &grid, rake_angle, clearance_angle, fillet, speed, feed);
    //particle_gpu* particles = setup_ref_cut_FEM_tool_textured(ny, &grid, rake_angle, clearance_angle, fillet, speed, feed);
	//printf("Cutting simulation, using FEM tool \n");
#else
	particle_gpu *particles = setup_ref_cut(ny, &grid, rake_angle, clearance_angle, fillet, speed, feed);
	printf("Cutting simulation, using SPH tool \n");
#endif

	//preliminary benchmarks
//	particle_gpu *particles = setup_rings(80, &grid);
//	particle_gpu *particles = setup_impact(20, &grid);

	check_cuda_error("init\n");

	if ( global_tool == 0) {
		global_tool = new tool();
	}
	
#ifdef USE_FEM_TOOL
	leap_frog *stepper = new leap_frog(particles->N, grid->num_cell(), global_tool_FEM->node_number);
#else
	leap_frog *stepper = new leap_frog(particles->N, grid->num_cell());
#endif
	//make results directoy if not present
	struct stat st = {0};
	if (stat("results", &st) == -1) {
		//mkdir("results", 0777);
		std::system("mkdir results");
	}


	//clear files from result directory
	int ret;
	
	std::system("mkdir results");

	cudaEvent_t start, stop, intermediate;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&intermediate);
	cudaEventRecord(start);

	int last_step = global_t_final/global_dt;
	int freq = last_step/100;
	int freq_remesh = freq / 20;
	printf("freq_remesh is %d\n", freq_remesh);
	float_t time_btwn_dumps = freq*global_dt;
	int report_time_at = 0.1*last_step;	//print expected runtime at 10% of sim time
#ifdef USE_FEM_TOOL 
	float2_t tl_ref;
	tl_ref.x = global_tool_FEM->tl_point_ref.x;
	tl_ref.y = global_tool_FEM->tl_point_ref.y;
#endif
	//log files for tool wear and forces
	char buf_wear[256];
	sprintf(buf_wear, "results/wear_%f_%f", speed, feed);

	char buf_force[256];
	sprintf(buf_force, "results/force_%f_%f", fillet, feed);

	unsigned int print_iter = 0;
	FILE *fp_tool_force = fopen(buf_force, "w+");
	FILE *fp_tool_wear = fopen(buf_wear, "w+");

	for (int step = 0; step < last_step+1; step++) {

		if (step == report_time_at) {
			cudaEventRecord(intermediate);
			cudaEventSynchronize(intermediate);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, intermediate);

			float_t seconds_so_far = milliseconds/1e3;
			float_t percent_done = 100*step/((float_t) last_step);
			float_t time_left = seconds_so_far/percent_done*100;
			printf("Seconds so far: %f\n", seconds_so_far);
			printf("estimated time: %f seconds, which is %f hours\n", time_left, time_left/60./60.);
		}

		// dump vtk file
		// report time left
		// log forces and tool wear if configured

		float_t percent_done = 100 * step/((float_t) last_step);

		if (percent_done > 35.2941 && wear_pro == false) wear_pro = true; //11.75, 28.5714, 23.5294, 35.2941， 81.4815, 68., 62.96296,  53.125

		if (step % freq == 0) {

			cudaEventRecord(intermediate);
			cudaEventSynchronize(intermediate);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, intermediate);

			float_t seconds_so_far = milliseconds/1e3;
			float_t percent_done = 100 * step/((float_t) last_step);
			float_t time_left = seconds_so_far/percent_done*100;

			//if (percent_done >9.08) wear_pro = true;
			//int temp = poll_temp();
			poll_memory_usage();

			printf("%06d of %06d: %02.1f percent done, %f of est runtime %f\n",
					step, last_step, percent_done, seconds_so_far, time_left);

			vtk_writer_write(particles, print_iter);
			if (global_tool) {
#ifdef USE_FEM_TOOL 

				vtk_writer_write_FEM_tool(global_tool_FEM, print_iter);
				vtk_writer_write_tool_org(global_tool_FEM, print_iter, tl_ref);

#ifdef WEAR_NODE_SHIFT
				vtk_writer_write_FEM_segments(global_tool_FEM, print_iter);
				//vtk_writer_write_contact_variables(global_tool_FEM, print_iter);
#endif

#else
				vtk_writer_write(global_tool, print_iter);
#endif	

				float2_t force_tool = report_force(particles);
				fprintf(fp_tool_force, "%.9g %.9g\n", force_tool.x, force_tool.y);
				fflush(fp_tool_force);
			}
			check_cuda_error();

			if (global_wear != 0 && log_wear) {
				float_t wear_usui_avg, wear_usui_min, wear_usui_max;
				global_wear->eval_usui(particles, time_btwn_dumps, wear_usui_min, wear_usui_max, wear_usui_avg);
				fprintf(fp_tool_wear, "%e %e %e %e\n", wear_usui_min, wear_usui_max, wear_usui_avg, global_wear->get_accum_wear());
				fflush(fp_tool_wear);
			}

			print_iter++;
		}
#ifdef WEAR_NODE_SHIFT
			    if (wear_pro  && step % freq_remesh == 0){
#ifdef REMESH_GMSH
                remesh_gmsh(global_tool_FEM, freq_remesh);
#else
			    remesh_low_level(global_tool_FEM, freq);
#endif
			    }	
#endif

		//exectue time step
#ifdef USE_FEM_TOOL
		stepper->step(particles, grid, global_tool_FEM);
#else
		stepper->step(particles, grid);
#endif
		global_step++;	

	}
	fclose(fp_tool_force);
	fclose(fp_tool_wear);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("gpu time: (seconds) %f\n", milliseconds/1e3);

	return 0;
}