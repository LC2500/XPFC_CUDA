#include "globals.cuh"

double mean(double tmp_sum) {
        int i, j;
		
        // Free energy calculation <start>
        // #pragma omp parallel for reduction(+:tmp_sum)

        for (i = 0; i < Mx; i++) {
                for (j = 0; j < My; j++){
			tmp_sum += psi_host[i*My + j].x;	
		}
	}

	tmp_sum = tmp_sum / (double)M2;
	//cout << "tmp_sum = " << tmp_sum << endl;
	return tmp_sum;
}


void configure_avg_density( void ) {
	double psi_diff = 0.0; // If the .vtk average density is not the same as the desired density
	int i, j;
	
	printf("\n######## Configure Average Density ######## \n");
	psi0_vtk = mean(psi0_vtk);
	printf("psi_0 = %lf\n", psi_0);	
	printf("psi0_vtk = %lf\n", psi0_vtk);

        if (psi0_vtk != psi_0) {
                psi_diff = psi_0 - psi0_vtk;
        }
	printf("psi_diff = psi0_vtk-psi_0 = %lf\n", psi_diff);

	for(i=0; i<Mx; i++) {
		for(j=0; j<My; j++){
			psi_host[i*My+j].x += psi_diff;
		}	
	}
	printf("######## Configure Average Density Complete ######## \n\n");


}


