#include <petsc.h>

static char help[]="poisson solver\n\n\n";

PetscErrorCode formMatrix(DM da, Mat A);
PetscErrorCode formExact(DM da, Vec uexact);
PetscErrorCode formRHS(DM da, Vec b);



int main(int argc, char **argv){
	PetscErrorCode     ierr;
	DM                 da;
	Mat                A;
	Vec                b, u, uexact;
	KSP                ksp;
	PetscReal          errnorm;
	DMDALocalInfo      info;

	ierr = PetscInitialize(&argc, &argv, (char*) 0, help); CHKERRQ(ierr);
	
	// change default 9x9 size using -da_grid_x M -da_grid_y N, or -da_refine
	DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 9, 9, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da);
	
	// create linear system matrix A
	DMSetFromOptions(da);
	DMSetUp(da);
	DMCreateMatrix(da, &A);
	MatSetFromOptions(A);

	// create RHS b, approx solution u, exact solution uexact
	DMCreateGlobalVector(da, &b);
	VecDuplicate(b, &u);
	VecDuplicate(b, &uexact);

	// fill vectors and assemble linear systems
	formExact(da, uexact);
	formRHS(da, b);
	formMatrix(da, A);

	// create and solve the linear system
	KSPCreate(PETSC_COMM_WORLD, &ksp);
	KSPSetOperators(ksp, A, A);
	KSPSetFromOptions(ksp);
	KSPSolve(ksp, b, u);

	// report on grid and numericall error
	VecAXPY(u, -1.0, uexact);
	VecNorm(u, NORM_INFINITY, &errnorm);
	DMDAGetLocalInfo(da, &info);
	PetscPrintf(PETSC_COMM_WORLD, "on %d x %d grid: error |u-uexact|_inf=%g\n", info.mx, info.my, errnorm);

	VecDestroy(&u); 
	VecDestroy(&uexact);
	VecDestroy(&b);
	MatDestroy(&A); KSPDestroy(&ksp); DMDestroy(&da);

	return PetscFinalize();
}



PetscErrorCode formMatrix(DM da, Mat A){
	DMDALocalInfo        info;
	MatStencil           row, col[5];
	PetscReal            hx, hy, v[5];
	PetscInt             i, j, ncols;

	DMDAGetLocalInfo(da, &info);
	hx = 1.0/(info.mx-1);
	hy = 1.0/(info.my-1);

	for(j=info.ys; j<info.ys+info.ym; j++){
		for(i=info.xs; i<info.xs + info.xm; i++){
			row.j = j;
			row.i = i;
			col[0].i = i;
			col[0].j = j;
			ncols = 1;
			if(i==0 || i==info.mx-1 || j==0 || j==info.my-1){
				v[0] = 1.0;
			}else{
				v[0] = 2*(hy/hx + hx/hy);
				if(i-1>0){
					col[ncols].j = j;
					col[ncols].i = i-1;
					v[ncols++] = -hy/hx;
				}
				if(i+1<info.mx-1){
					col[ncols].j = j;
					col[ncols].i = i+1;
					v[ncols++] = -hy/hx;
				}
				if(j-1>0){
					col[ncols].j = j-1;
					col[ncols].i = i;
					v[ncols++] = -hx/hy;
				}
				if(j+1<info.my-1){
					col[ncols].j = j+1;
					col[ncols].i = i;
					v[ncols++] = -hx/hy;
				}
			}
			MatSetValuesStencil(A, 1, &row, ncols, col, v, INSERT_VALUES);
		}
	}
	MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
	return 0;
}




PetscErrorCode formExact(DM da, Vec uexact){
	PetscInt        i, j;
	PetscReal       hx, hy, x, y, **auexact;
	DMDALocalInfo   info;

	DMDAGetLocalInfo(da, &info);
	hx = 1.0/(info.mx-1);
	hy = 1.0/(info.my-1);
	DMDAVecGetArray(da, uexact, &auexact);
	for(j=info.ys; j<info.ys+info.ym; j++){
		y=j*hy;
		for(i=info.xs; i<info.xs+info.xm; i++){
			x=i*hx;
			auexact[j][i] = x*x * (1.0 - x*x) * y*y * (y*y - 1.0);
		}
	}
	DMDAVecRestoreArray(da, uexact, &auexact);
	return 0;
}




PetscErrorCode formRHS(DM da, Vec b){
	PetscInt       i, j;
	PetscReal      hx, hy, x, y, f, **ab;
	DMDALocalInfo  info;

	DMDAGetLocalInfo(da, &info);
	hx = 1.0/(info.mx - 1);
	hy = 1.0/(info.my - 1);

	DMDAVecGetArray(da, b, &ab);
	for(j=info.ys; j<info.ys+info.ym; j++){
		y = j*hy;
		for(i=info.xs; i<info.xs + info.xm; i++){
			x = i*hx;

			if(i==0 || i==info.mx-1 || j==0 || j==info.my-1){
				ab[j][i] = 0.0;
			} else {
				f = 2.0 * ( (1.0 - 6.0*x*x) * y*y * (1.0 - y*y) + (1.0 - 6.0*y*y)* x*x * (1.0 - x*x));
				ab[j][i] = hx*hy*f;
			}
		}
	}
	DMDAVecRestoreArray(da, b, &ab);
	return 0;
}
