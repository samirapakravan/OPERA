#include <petsc.h>

static char help[] = "Newton's methodfor a two variable system.\n no analytical Jacobian, run with -snes_fd or -snes_mg. \n\n\n";


extern PetscErrorCode FormFunction(SNES, Vec, Vec, void*);
extern PetscErrorCode FormJacobian(SNES snes, Vec x, Mat J, Mat P, void *ctx);

// a struct to pass parameters to the FormFunction(.., void*)
typedef struct {
	PetscReal b;
} AppCtx;


int main(int argc, char **argv){
	SNES snes;
	Vec x, r;
	Mat J;
	AppCtx user;
	user.b = 2.0;
	
	PetscInitialize(&argc, &argv, NULL, help);
	
	VecCreate(PETSC_COMM_WORLD, &x);
	VecSetSizes(x, PETSC_DECIDE, 2);
	VecSetFromOptions(x);
	VecSet(x, 1.0);
	VecDuplicate(x, &r);

	MatCreate(PETSC_COMM_WORLD, &J);
	MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, 2, 2);
	MatSetFromOptions(J);
	MatSetUp(J);

	SNESCreate(PETSC_COMM_WORLD, &snes);
	SNESSetFunction(snes, r, FormFunction, &user);
	SNESSetJacobian(snes, J, J, FormJacobian, &user);
	SNESSetFromOptions(snes);

	SNESSolve(snes, NULL, x);
	
	VecView(x, PETSC_VIEWER_STDOUT_WORLD);

	VecDestroy(&x); 
	VecDestroy(&r);
	MatDestroy(&J);
	SNESDestroy(&snes);
	return PetscFinalize();	
}

PetscErrorCode FormFunction(SNES snes, Vec x, Vec F, void *ctx){
	AppCtx  *user = (AppCtx*) ctx; // cast to struct pointer from void pointer
	const PetscReal b = user->b, *ax;
	PetscReal *aF;
	
	VecGetArrayRead(x, &ax);
	VecGetArray(F, &aF);
	
	aF[0] = (1.0/b) * PetscExpReal(b * ax[0]) - ax[1];
	aF[1] = ax[0] * ax[0] + ax[1] * ax[1] - 1.0;

	VecRestoreArray(F, &aF);
	VecRestoreArrayRead(x, &ax);
	return 0;
}

PetscErrorCode FormJacobian(SNES snes, Vec x, Mat J, Mat P, void *ctx){
	AppCtx     *user = (AppCtx*) ctx;
	const PetscReal b = user->b, *ax;
	PetscReal       v[4];
	PetscInt        row[2] = {0,1}, col[2] = {0,1};

	VecGetArrayRead(x, &ax);
	v[0] = PetscExpReal(b * ax[0]); v[1] = -1.0;
	v[2] = 2.0 * ax[0];             v[3] = 2.0 * ax[1];
	VecRestoreArrayRead(x, &ax);
	MatSetValues(P, 2, row, 2, col, v, INSERT_VALUES);
	MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY);

	if(J != P) {
		MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);
	}
	return 0;
}
