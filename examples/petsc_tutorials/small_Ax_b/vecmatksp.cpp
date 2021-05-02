#include <petsc.h>

static char help[] = "Solve 4x4 linear system using KSP.\n";

int main(int argc, char **argv){
	Vec      x, b;
	Mat      A;
	KSP      ksp;
	PetscInt i, j[4]={0, 1, 2, 3};
	PetscReal ab[4] = {7.0, 1.0, 1.0, 3.0},
		  aA[4][4] = {{1.0, 2.0, 3.0, 0.0},
		              {2.0, 1.0, -2.0, -3.0},
		              {-1.0, 1.0, 1.0, 0.0},
		              {0.0, 1.0, 1.0, -1.0}};

	PetscInitialize(&argc, &argv, NULL, help);

	VecCreate(PETSC_COMM_WORLD, &b);
	VecSetSizes(b, PETSC_DECIDE, 4);
	VecSetFromOptions(b);
	VecSetValues(b, 4, j, ab, INSERT_VALUES);
	VecAssemblyBegin(b);
	VecAssemblyEnd(b);

	MatCreate(PETSC_COMM_WORLD, &A);
	MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 4, 4);
	MatSetFromOptions(A);
	MatSetUp(A);
	for(i=0; i<4; i++)
		MatSetValues(A, 1, &i, 4, j, aA[i], INSERT_VALUES);
	MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

	KSPCreate(PETSC_COMM_WORLD, &ksp);
	KSPSetOperators(ksp, A, A);
	KSPSetFromOptions(ksp);
	VecDuplicate(b, &x);
	KSPSolve(ksp, b, x);
	VecView(x, PETSC_VIEWER_STDOUT_WORLD);

	KSPDestroy(&ksp); 
	MatDestroy(&A);
        VecDestroy(&b); 
	VecDestroy(&x);
	return PetscFinalize();	
}
