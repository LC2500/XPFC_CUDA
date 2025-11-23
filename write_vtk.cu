#include "globals.cuh"


void writeVTKFile(Complex* densityData, int nx, int ny, int Lx, int Ly, int t_i) {
    
    string vtkFilename = "density_data" + to_string(t_i) + ".vtk"; 
    // Open the VTK file for writing
    ofstream vtkFile(vtkFilename);
    // ofstream vtkFile(vtkFilename, std::ios::out | std::ios::binary);

    // Write the header for the VTK file
    vtkFile << "# vtk DataFile Version 3.0\n";
    vtkFile << "Density Data\n";
    vtkFile << "ASCII\n";
    vtkFile << "DATASET STRUCTURED_POINTS\n";
    vtkFile << "DIMENSIONS " << nx << " " << ny << " 1\n";
    vtkFile << "ORIGIN 0 0 0\n";
    vtkFile << "SPACING " << double(Lx)/double(nx) << " " << double(Ly)/double(ny) << " 1\n";
    vtkFile << "POINT_DATA " << nx * ny << "\n";
    vtkFile << "SCALARS DensityData float\n";
    vtkFile << "LOOKUP_TABLE default\n";

    
    // Write the density data
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
                vtkFile << densityData[j*ny + i].x << "\n";
        }
    }
    
    // Close the VTK file
    vtkFile.close();

    // cout << "VTK file generated successfully.\n";
}

