#include <petsc.h>

static char help[] = "Newton's methodfor a two variable system.\n no analytical Jacobian, run with -snes_fd or -snes_mg. \n\n\n";


extern PetscErrorCode FormFunction(SNES, Vec, Vec, void*);


int main(int argc, char **argv){
	SNES snes;
	Vec x, r;
	
	PetscInitialize(&argc, &argv, NULL, help);
	VecCreate(PETSC_COMM_WORLD, &x);
	VecSetSizes(x, PETSC_DECIDE, 2);
	VecSetFromOptions(x);
	VecSet(x, 1.0);
	VecDuplicate(x, &r);

	SNESCreate(PETSC_COMM_WORLD, &snes);
	SNESSetFunction(snes, r, FormFunction, NULL);
	SNESSetFromOptions(snes);
	SNESSolve(snes, NULL, x);
	VecView(x, PETSC_VIEWER_STDOUT_WORLD);

	VecDestroy(&x); 
	VecDestroy(&r);
	SNESDestroy(&snes);
	return PetscFinalize();	
}

PetscErrorCode FormFunction(SNES snes, Vec x, Vec F, void *ctx){
	const PetscReal b = 2.0, *ax;
	PetscReal *aF;
	
	VecGetArrayRead(x, &ax);
	VecGetArray(F, &aF);
	
	aF[0] = (1.0/b) * PetscExpReal(b * ax[0]) - ax[1];
	aF[1] = ax[0] * ax[0] + ax[1] * ax[1] - 1.0;

	VecRestoreArray(F, &aF);
	VecRestoreArrayRead(x, &ax);
	return 0;
}
