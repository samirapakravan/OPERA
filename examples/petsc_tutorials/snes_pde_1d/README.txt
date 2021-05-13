./executable -snes_test_jacobian
./executable -snes_converged_reason -da_refine 6 -snes_monitor
for K in 0 1 2 3 4 5 6; do ./executable -da_refine $K; done
./executable -da_refine 6 -snes_monitor_solution draw -draw_pause 1

Use coloring if using finite difference for Jacobian:
./executable -snes_converged_reason -da_refine 10 -snes_fd_color
this is actually very efficient (13 evaluations of F in total, compared to 16390 calls for -snes_fd, the analytical Jacobian implemented here needs 3 function calls). To see the number of evaluations use -log_view | grep Eval with any of these commands.

If using the JFNK (Jacobian free) solver with the option -snes_mf, it usually does not do well without preconditioning. Specially for larger problems (increasing resolution) convergence is lost due to the increased number of iterations of the KSP solvers. Instead one should use the -snes_mf_operator option with an approximation for the Jacobian matrix implmented instead:

./executable -snes_converged_reason -da_refine 10 -snes_mf_operator -rct_noRinJ -snes_monitor

This is achieved in the current example by only keeping the tridiagonal leading-order part of the Jacobian (by enabling -rct_noRinJ option that approximates the Jacobian in the code). This approximation should preseve most spectral characteristics of the Jacobian.

Also one can select the linesearch method in each Newton step from -snes_linesearch_type bt(defalt)/cp/l2. For example:
./executable -snes_converged_reason -da_refine 10 -snes_monitor -snes_linesearch_type cp
