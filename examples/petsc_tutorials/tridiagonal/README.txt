mpirun --oversubscribe -np 10 ./executable -tri_m 5 -ksp_monitor -a_mat_view ::ascii_dense

./executable -help | grep tri_

time ./executable -tri_m 10000 -ksp_type gmres -pc_type ilu -ksp_monitor
