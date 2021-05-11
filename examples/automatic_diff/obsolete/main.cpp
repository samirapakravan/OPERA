#include <boost/math/differentiation/autodiff.hpp>
#include <iostream>


template <typename T> 
T fourth_power(T const& x){
	T x4 = x*x;
	x4 *= x4;
	return x4;
}

template <typename T>
T func(T const& x, T const& y){
	return x*y;
}

int main(int argc, char **argv){
	using namespace boost::math::differentiation;

	constexpr unsigned Order = 5;
	
	auto const x = make_fvar<double, Order>(2.0);
	//auto const y = make_fvar<double, Order>(3.0);
	auto const f = fourth_power(x);

	for(unsigned int i=0; i<=Order; ++i)
		std::cout<< "f.derivative("<< i << ") = " << f.derivative(i) << std::endl;


	return 0;
}
