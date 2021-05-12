./executable -snes_fd -snes_monitor -snes_rtol 1e-9 -snes_converged_reason
 ./executable -log_view | grep Eval
 ./executable -snes_test_jacobian
