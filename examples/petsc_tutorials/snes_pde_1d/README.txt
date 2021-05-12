./executable -snes_test_jacobian
./executable -snes_converged_reason -da_refine 6 -snes_monitor
for K in 0 1 2 3 4 5 6; do ./executable -da_refine $K; done
./executable -da_refine 6 -snes_monitor_solution draw -draw_pause 1

