// Wrapper of the GSL interpolation functions that lets you define a
// 1D function by points and then query points on that function.
//
// Author: Mark Desnoyer (mdesnoyer@gmail.com)
// Date: Oct 2012

#ifndef __INTERPOLATION_H__
#define __INTERPOLATION_H__
#include <vector>
#include <gsl/gsl_spline.h>
#include <assert.h>
#include <algorithm>
#include <limits>
#include <exception>

template <typename T>
class SortOrder {
public:
  SortOrder(const std::vector<T>& sortArray) : sortArray_(sortArray) {}

  bool operator()(int lhs, int rhs) const {
    return sortArray_[lhs] < sortArray_[rhs];
  }

private:
  const std::vector<T>& sortArray_;
};

// Classes that inherit this abstract class must intialize spline_ in
// their constructors.
class Interpolator {
public:
  class out_of_bounds : public std::exception {};

  template <typename InputIterator>
  Interpolator(InputIterator xFirst, InputIterator xLast,
               InputIterator yFirst, InputIterator yLast)
    : x_(), y_(), acc_(gsl_interp_accel_alloc()) {
    std::vector<double> x(xFirst, xLast);
    std::vector<double> y(yFirst, yLast);
    std::vector<int> idx;
    for (unsigned int i = 0u; i < x.size(); ++i) {
      idx.push_back(i);
    }

    assert(x.size() == y.size());
    
    // Get the sorting indicies based on the x matrix
    std::sort(idx.begin(), idx.end(), SortOrder<double>(x));

    // Populate the data in ascending x order skipping equal values
    double curX = std::numeric_limits<double>::quiet_NaN();
    for (unsigned int i = 0u; i < idx.size(); ++i) {
      if (curX != x[idx[i]]) {
        x_.push_back(x[idx[i]]);
        y_.push_back(y[idx[i]]);
      }
      curX = x[idx[i]];
    }

  }

  virtual ~Interpolator();                   

  // Evaluate the interpolation at xi
  double operator()(double xi) const {
    if (xi < x_[0] || xi > x_.back()) {
      throw out_of_bounds();
    }
    return gsl_spline_eval(spline_, xi, acc_);
  }

  // Evaluate the derivative at xi
  double EvalDeriv(double xi) const {
    if (xi < x_[0] || xi > x_.back()) {
      throw out_of_bounds();
    }
    return gsl_spline_eval_deriv(spline_, xi, acc_);
  }

  double minY() const { return y_.front(); }
  double maxY() const { return y_.back(); }

protected:
  std::vector<double> x_;
  std::vector<double> y_;

  gsl_interp_accel* acc_;
  gsl_spline* spline_;

};

class LinearInterpolator : public Interpolator {
public:
  LinearInterpolator(const std::vector<double>& x, const std::vector<double>& y)
    : Interpolator(x.begin(), x.end(), y.begin(), y.end()) {
    InitSpline();
  }

  template <typename InputIterator>
  LinearInterpolator(InputIterator xFirst, InputIterator xLast,
                     InputIterator yFirst, InputIterator yLast)
    : Interpolator(xFirst, xLast, yFirst, yLast) {
    InitSpline();
  }

  virtual ~LinearInterpolator();

private:
  void InitSpline();

};

class SplineInterpolator : public Interpolator {
public:
  SplineInterpolator(const std::vector<double>& x, const std::vector<double>& y)
    : Interpolator(x.begin(), x.end(), y.begin(), y.end()) {
    InitSpline();
  }

  template <typename InputIterator>
  SplineInterpolator(InputIterator xFirst, InputIterator xLast,
                     InputIterator yFirst, InputIterator yLast)
    : Interpolator(xFirst, xLast, yFirst, yLast) {
    InitSpline();
  }

  virtual ~SplineInterpolator();

private:
  void InitSpline();

};

#endif // __INTERPOLATION_H__
