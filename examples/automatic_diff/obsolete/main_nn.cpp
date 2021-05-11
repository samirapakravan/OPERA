static char help[] = "hello neural network!\n";
#include <iostream>
#include <petsc.h>





//---------------------------------------------------------------------------
/* autodiff include */
#include <autodiff/reverse.hpp>
using namespace autodiff;


// The multi-variable function for which derivatives are needed
var f(var x, var y, var z)
{
    return 1 + x + y + z; // + x*y + y*z + x*z + x*y*z + exp(x/y + y/z);
}



//---------------------------------------------------------------------------
// boost automatic differentiation 
#include <boost/math/differentiation/autodiff.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
using namespace boost::math::differentiation;


template <typename W, typename X, typename Y, typename Z>
promote<W, X, Y, Z> f(const W& w, const X& x, const Y& y, const Z& z) {
  using namespace std;
  return exp(w * sin(x * log(y) / z) + sqrt(w * z / (x * y))) + w * w / tan(z);
}
//--------------------------------------------------------------------------

/* main */

int main(int argc, char **argv) {
  PetscErrorCode     ierr;
  Vec                b;
  PetscReal ab[4] = {11.0, 12.0, 13.0, 14.0};
  PetscInt i, j[4]={0, 1, 2, 3};
  PetscRandom    rand;
/*
  var x = 1.0;         // the input variable x
  var y = 2.0;         // the input variable y
  var z = 3.0;         // the input variable z
  var V[3] = {x,y,z};
  var u = f(x, y, z);  // the output variable u
  auto [ux, uy, uz] = derivatives(u, wrt(x, y, z)); // evaluate the derivatives of u with respect to x, y, z
  std::cout << "u = "  << u << std::endl;    // print the evaluated output u
  std::cout << "ux = " << ux << std::endl;  // print the evaluated derivative ux
  std::cout << "uy = " << uy << std::endl;  // print the evaluated derivative uy
  std::cout << "uz = " << uz << std::endl;  // print the evaluated derivative uz
*/

  ierr = PetscInitialize(&argc, &argv, (char*) 0, help); CHKERRQ(ierr);

  VecCreate(PETSC_COMM_WORLD, &b);
  VecSetSizes(b, PETSC_DECIDE, 3);
  VecSetFromOptions(b);
  
  PetscRandomCreate(PETSC_COMM_WORLD,&rand);
  PetscRandomSetFromOptions(rand);


  PetscInt nloc;
  VecGetLocalSize(b,&nloc);
  var Vs[nloc];

  double *vec_p;
  VecGetArray(b, &vec_p);
  for (i=0; i<nloc; i++){
	  PetscScalar val;
	  PetscRandomGetValue(rand,&val);
	  Vs[i] = val;
	  vec_p[i] = val;
  }
  VecRestoreArray(b, &vec_p);

  var u = f(Vs[0], Vs[1], Vs[2]);  // the output variable u
  auto [ux, uy, uz] = derivatives(u, wrt(Vs[0], Vs[1], Vs[2])); // evaluate the derivatives of u with respect to x, y, z
  std::cout << "u = "  << u << std::endl;    // print the evaluated output u
  std::cout << "ux = " << ux << std::endl;  // print the evaluated derivative ux
  std::cout << "uy = " << uy << std::endl;  // print the evaluated derivative uy
  std::cout << "uz = " << uz << std::endl;  // print the evaluated derivative uz

/*
  using float50 = boost::multiprecision::cpp_bin_float_50;
  // Declare 4 independent variables together into a std::tuple.
  auto const variables = make_ftuple<float50, 1, 1, 1, 1>(ab[0], ab[1], ab[2], ab[3]);//(11, 12, 13, 14);
  auto const& w = std::get<0>(variables);
  auto const& x = std::get<1>(variables); 
  auto const& y = std::get<2>(variables); 
  auto const& z = std::get<3>(variables);  
  auto const v = f(w, x, y, z);
  // Calculated from Mathematica symbolic differentiation.
  std::cout << v.derivative(1, 1, 1, 1) << '\n';
*/


  VecDestroy(&b);
  PetscRandomDestroy(&rand);
  return PetscFinalize();
}

