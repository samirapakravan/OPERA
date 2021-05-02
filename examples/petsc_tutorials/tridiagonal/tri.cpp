static char help[] = "Solve a tridiagonal system of arbitrary size m.\n Option prefix = tri_ \n";

#include <petsc.h>

int main(int argc, char **argv){
	Vec       x, b, xexact;
	Mat       A;
	KSP       ksp;
	PetscInt  m = 4, i, lstart, lend, j[3];
	PetscReal v[3], xval, errnorm;

	PetscInitialize(&argc,&argv,NULL,help);

	PetscOptionsBegin(PETSC_COMM_WORLD,"tri_","Options for tri", "");
	PetscOptionsInt("-m", "dimension of linear system", "tri.cpp", m, &m, NULL);
	PetscOptionsEnd();

	VecCreate(PETSC_COMM_WORLD, &x);
	VecSetSizes(x, PETSC_DECIDE, m);
	VecSetFromOptions(x);
	VecDuplicate(x, &b);
	VecDuplicate(x, &xexact);

	MatCreate(PETSC_COMM_WORLD, &A);
	MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m, m);
	MatSetOptionsPrefix(A, "a_");
	MatSetFromOptions(A);
	MatSetUp(A);
	MatGetOwnershipRange(A, &lstart, &lend);
	for(i=lstart; i<lend; i++){
		if(i==0){
			v[0] =  3.0;
			v[1] = -1.0;
			j[0] =  0;
			j[1] =  1;
			MatSetValues(A, 1, &i, 2, j, v, INSERT_VALUES);
		} else {
			v[0] = -1.0;
			v[1] =  3.0;
			v[2] = -1.0;
			j[0] = i-1;
			j[1] = i;
			j[2] = i+1;
			if(i == m-1){
				MatSetValues(A, 1, &i, 2, j, v, INSERT_VALUES);
			} else {
				MatSetValues(A, 1, &i, 3, j, v, INSERT_VALUES);
			}
		}
		xval = PetscExpReal(PetscCosReal(i));
		VecSetValues(xexact, 1, &i, &xval, INSERT_VALUES);
	}
	MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
	VecAssemblyBegin(xexact);
	VecAssemblyEnd(xexact);
	MatMult(A, xexact, b);

	KSPCreate(PETSC_COMM_WORLD, &ksp);
	KSPSetOperators(ksp, A, A);
	KSPSetFromOptions(ksp);
	KSPSolve(ksp, b, x);

	VecAXPY(x, -1.0, xexact);
	VecNorm(x, NORM_2, &errnorm);
	PetscPrintf(PETSC_COMM_WORLD, "error for m=%d system is |x - xexact|_2=%.1e\n", m, errnorm);

	KSPDestroy(&ksp);
	MatDestroy(&A);
	VecDestroy(&x);
	VecDestroy(&b);
	VecDestroy(&xexact);

	return PetscFinalize();
	
}
