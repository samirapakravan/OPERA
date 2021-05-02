if on a local machine and number of openmpi slots is limited, then try oversubscribing the number of slots by using the --oversubscribe flag. 
For example on a machine with 2 physical cores, you can run on 20 processors by issuing:
$ mpirun --oversubscribe -np 20 executable


Alternatively, create a "hostfile" with the content: "localhost slots=20", and run the mpirun with this file:
$ mpirun --hostfile hostfile -np 20 executable
