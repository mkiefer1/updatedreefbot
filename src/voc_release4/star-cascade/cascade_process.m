function [det, all] = cascade_process(image, csc_model)

% bbox = cascade_process(image, csc_model)
% Detect objects that score above a threshold, return bonding boxes.
% If the threshold is not included we use the one in the model.
% This should lead to high-recall but low precision.

pyra = featpyramid(image, csc_model);
[det, all] = cascade_detect(pyra, csc_model, csc_model.thresh);

if ~isempty(det)
%   try
%     % attempt to use bounding box prediction, if available
%     bboxpred = csc_model.bboxpred;
%     [det all] = clipboxes(image, det, all);
%     [det all] = bboxpred_get(bboxpred, det, all);
%   catch
%     warning('no bounding box predictor found');
%   end
  [det all] = clipboxes(image, det, all);
  I = nms(det, 0.5);
  det = det(I,:);
  all = all(I,:);
end
