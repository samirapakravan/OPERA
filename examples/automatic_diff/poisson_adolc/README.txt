./executable -da_grid_x 9 -da_grid_y 9 -ksp_monitor -ksp_monitor_solution draw -draw_pause 3
./executable -da_grid_x 9 -da_grid_y 9 -ksp_view_mat draw -draw_pause 10
./executable -da_grid_x 9 -da_grid_y 9 -ksp_monitor -ksp_view_solution binary:u.dat
./executable -da_refine 2 -ksp_monitor -ksp_view_solution draw -draw_pause 3
./executable -da_grid_x 9 -da_grid_y 9 -dm_view draw -draw_pause 3
