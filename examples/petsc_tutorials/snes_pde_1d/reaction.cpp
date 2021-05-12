#include <petsc.h>

static char help[]="nonlinear reaction-diffusion PDE in 1D.\n\n\n";


typedef struct{
	PetscReal rho, M, alpha, beta;
	PetscBool noRinJ;
} AppCtx;

extern PetscErrorCode InitialAndExact(DMDALocalInfo*, PetscReal*, PetscReal*, AppCtx*);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo *, PetscReal *, PetscReal *, AppCtx *);
extern PetscErrorCode FormJacobianLocal(DMDALocalInfo *, PetscReal *, Mat, Mat, AppCtx*);
extern PetscReal f_source(PetscReal);

int main(int argc, char **argv){
	DM            da;
	SNES          snes;
	AppCtx        user;
	Vec           u, uexact;
	PetscReal     errnorm, *au, *auex;
	DMDALocalInfo info;

	PetscInitialize(&argc, &argv, NULL, help);

	user.rho    = 10.0;
	user.M      = PetscSqr(user.rho / 12.0);
        user.alpha  = user.M;
	user.beta   = 16.0 * user.M;
	user.noRinJ = PETSC_FALSE;

	PetscOptionsBegin(PETSC_COMM_WORLD, "rct_", "options for reaction", "");
	PetscOptionsBool("-noRinJ", "do not include R(u) term in Jacobian", "reaction.c", user.noRinJ, &(user.noRinJ), NULL);
	PetscOptionsEnd();

	DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 9, 1, 1, NULL, &da);
	DMSetFromOptions(da);
	DMSetUp(da);
	DMSetApplicationContext(da, &user);

	DMCreateGlobalVector(da, &u);
	VecDuplicate(u, &uexact);
	DMDAVecGetArray(da, u, &au);

	DMDAGetLocalInfo(da, &info);
	DMDAVecGetArray(da, uexact, &auex);
	InitialAndExact(&info, au, auex, &user);
	DMDAVecRestoreArray(da, u, &au);
	DMDAVecRestoreArray(da, uexact, &auex);

	SNESCreate(PETSC_COMM_WORLD, &snes);
	SNESSetDM(snes, da);
	DMDASNESSetFunctionLocal(da, INSERT_VALUES, (DMDASNESFunction) FormFunctionLocal, &user);
	DMDASNESSetJacobianLocal(da, (DMDASNESJacobian) FormJacobianLocal, &user);
	SNESSetFromOptions(snes);
	
	SNESSolve(snes, NULL, u);
	
	VecAXPY(u, -1.0, uexact);
	VecNorm(u, NORM_INFINITY, &errnorm);
	PetscPrintf(PETSC_COMM_WORLD, "on %d point grid: |u-u_exact|_inf = %g\n", info.mx, errnorm);

	VecDestroy(&u); VecDestroy(&uexact);
	SNESDestroy(&snes); DMDestroy(&da);	

	return PetscFinalize();
}


PetscReal f_source(PetscReal x){
	return 0.0;
}


PetscErrorCode InitialAndExact(DMDALocalInfo *info, PetscReal *u0,
                               PetscReal *uex, AppCtx *user) {
    PetscInt   i;
    PetscReal  h = 1.0 / (info->mx-1), x;
    for (i=info->xs; i<info->xs+info->xm; i++) {
        x = h * i;
        u0[i]  = user->alpha * (1.0 - x) + user->beta * x;
        uex[i] = user->M * PetscPowReal(x + 1.0,4.0);
    }
    return 0;
}


PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal *u, PetscReal *FF, AppCtx *user){
	PetscInt i;
	PetscReal h = 1.0 / (info->mx - 1), x, R;

	for (i = info->xs; i < info->xs+info->xm; i++){
		if (i == 0) {                // left bc
			FF[i] = u[i] - user->alpha;
		} else if (i == info->mx-1){ // right bc
			FF[i] = u[i] - user->beta;
		} else {                     // interior points
			if (i==1){
				FF[i] = -u[i+1] + 2.0 * u[i] - user->alpha;
			} else if (i==info->mx-2) {
				FF[i] = -user->beta + 2 * u[i] - u[i-1];
			} else {
				FF[i] = -u[i+1] + 2.0 * u[i] - u[i-1];
			}
			R = -user->rho * PetscSqrtReal(u[i]);
			x = i * h;
			FF[i] -= h*h * (R + f_source(x));
		}
	}
	return 0;
}

PetscErrorCode FormJacobianLocal(DMDALocalInfo *info, PetscReal *u, Mat J, Mat P, AppCtx *user){
	PetscInt i, col[3];
	PetscReal h = 1.0 / (info->mx - 1), dRdu, v[3];
	for (i=info->xs; i<info->xs+info->xm; i++){
		if ((i==0) || (i==info->mx-1)) {
			v[0] = 1.0;
			MatSetValues(P, 1, &i, 1, &i, v, INSERT_VALUES);
		} else { 
			col[0] = i;
			v[0] = 2.0;
			if (!user->noRinJ) {
				dRdu = -(user->rho / 2.0) / PetscSqrtReal(u[i]);
				v[0] -= h*h *dRdu;
			}
			col[1] = i-1; v[1] = (i>1)          ? -1.0 : 0.0;
			col[2] = i+1; v[2] = (i<info->mx-2) ? -1.0 : 0.0;
			MatSetValues(P, 1, &i, 3, col, v, INSERT_VALUES);
		}
	}
	MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY);
	if (J != P){
		MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);
	}
	return 0;
}




