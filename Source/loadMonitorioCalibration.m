function cal = loadMonitorioCalibration(jsonPath)
% loadMonitorioCalibration  Read a Monitorio calibration JSON (v1) into a struct.
%
%   cal = loadMonitorioCalibration(jsonPath)
%
% Input
%   jsonPath : path to a JSON file produced by calibration/scripts/calibrate.py
%
% Output struct fields
%   bitXs                : 1xN row, integer X pixel positions (one per PD)
%   bitYs                : 1xN row, integer Y pixel positions
%   bitRadius            : scalar, max across per-PD bit radii
%   backgroundRadius     : scalar, max across per-PD background radii
%   channels             : 1xN cell array of DAQ channel names (e.g. 'Dev1/ai0')
%   perPdBitRadii        : 1xN row, per-PD bit-circle radii
%   perPdBackgroundRadii : 1xN row, per-PD black-background radii
%   monitor              : struct with .index / .width / .height
%   raw                  : the full jsondecode'd struct (for any extra fields)
%
% bitRadius and backgroundRadius are reduced to scalars (by max across PDs)
% because addVideoSyncTags currently uses a single radius for every bit.
% Per-PD values are retained in perPd* in case a future version wants
% to draw different-sized circles per bit.

arguments
    jsonPath (1, :) char
end

raw = jsondecode(fileread(jsonPath));

if ~isfield(raw, 'photodiodes') || isempty(raw.photodiodes)
    error('loadMonitorioCalibration:NoPDs', ...
          'Calibration file has no "photodiodes" entries: %s', jsonPath);
end

pds = raw.photodiodes;
n = numel(pds);

cal.bitXs = zeros(1, n);
cal.bitYs = zeros(1, n);
cal.perPdBitRadii = zeros(1, n);
cal.perPdBackgroundRadii = zeros(1, n);
cal.channels = cell(1, n);

for i = 1:n
    pd = pds(i);
    cal.bitXs(i) = round(pd.x_px);
    cal.bitYs(i) = round(pd.y_px);
    cal.perPdBitRadii(i) = pd.bit_radius_px;
    cal.perPdBackgroundRadii(i) = pd.background_radius_px;
    cal.channels{i} = pd.channel;
end

cal.bitRadius = max(cal.perPdBitRadii);
cal.backgroundRadius = max(cal.perPdBackgroundRadii);

if isfield(raw, 'monitor')
    cal.monitor = raw.monitor;
else
    cal.monitor = struct();
end

cal.raw = raw;
