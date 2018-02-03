#include "hog_detector/interpolation.h"
#include <gsl/gsl_interp.h>



Interpolator::~Interpolator() {
  if (spline_) {
    gsl_spline_free(spline_);
  }
  gsl_interp_accel_free(acc_);
}


LinearInterpolator::~LinearInterpolator() {
}

void LinearInterpolator::InitSpline() {
  spline_ = gsl_spline_alloc(gsl_interp_linear, x_.size());
  gsl_spline_init(spline_, &(x_[0]), &(y_[0]), x_.size());
}

SplineInterpolator::~SplineInterpolator() {
}

void SplineInterpolator::InitSpline() {
  spline_ = gsl_spline_alloc(gsl_interp_cspline, x_.size());
  gsl_spline_init(spline_, &(x_[0]), &(y_[0]), x_.size());
}
