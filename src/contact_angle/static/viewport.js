/* View boxes use original-image coordinates; stage sizes use CSS pixels. */
const DEFAULT_MIN_SCALE = 0.25;
const DEFAULT_MAX_SCALE = 8;
const FIT_MARGIN = 0.93;

const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));
const positive = (value) => Number.isFinite(value) && value > 0;

function dimensions(image, stage, limits) {
  if (!positive(image?.width) || !positive(image?.height)) throw new RangeError('Image dimensions must be positive and finite.');
  if (!positive(stage?.width) || !positive(stage?.height)) throw new RangeError('Stage dimensions must be positive and finite.');
  const minScale = limits.minScale ?? DEFAULT_MIN_SCALE;
  const maxScale = limits.maxScale ?? DEFAULT_MAX_SCALE;
  if (!positive(minScale) || !positive(maxScale) || maxScale < minScale) throw new RangeError('Invalid zoom limits.');
  return {minScale, maxScale};
}

function origin(center, extent, imageExtent) {
  // When an image fits along an axis, keep it centered on that axis. Otherwise
  // stop panning at its edges, so no empty space replaces visible image data.
  if (extent >= imageExtent) return (imageExtent - extent) / 2;
  return clamp(center - extent / 2, 0, imageExtent - extent);
}

export function constrainView(view, image, stage, limits = {}) {
  const {minScale, maxScale} = dimensions(image, stage, limits);
  const oldWidth = positive(view?.w) ? view.w : stage.width;
  const oldHeight = positive(view?.h) ? view.h : oldWidth * stage.height / stage.width;
  const scale = clamp(stage.width / oldWidth, minScale, maxScale);
  const w = stage.width / scale;
  const h = stage.height / scale;
  const proposedX = (Number.isFinite(view?.x) ? view.x : (image.width - oldWidth) / 2) + oldWidth / 2;
  const proposedY = (Number.isFinite(view?.y) ? view.y : (image.height - oldHeight) / 2) + oldHeight / 2;
  const centerX = Number.isFinite(proposedX) ? proposedX : image.width / 2;
  const centerY = Number.isFinite(proposedY) ? proposedY : image.height / 2;
  return {x: origin(centerX, w, image.width), y: origin(centerY, h, image.height), w, h};
}

export function zoomView(view, factor, anchor, image, stage, limits = {}) {
  const {minScale, maxScale} = dimensions(image, stage, limits);
  if (!positive(factor)) throw new RangeError('Zoom factor must be positive and finite.');
  const old = constrainView(view, image, stage, limits);
  const scale = clamp(stage.width / old.w * factor, minScale, maxScale);
  const w = stage.width / scale;
  const h = stage.height / scale;
  const ratio = w / old.w;
  const px = clamp(Number.isFinite(anchor?.x) ? anchor.x : old.x + old.w / 2, 0, image.width);
  const py = clamp(Number.isFinite(anchor?.y) ? anchor.y : old.y + old.h / 2, 0, image.height);
  // Use the effective ratio after clamping. Reusing the requested factor at
  // a zoom limit would move the image even though its magnification stopped.
  return constrainView({x: px - (px - old.x) * ratio, y: py - (py - old.y) * ratio, w, h}, image, stage, limits);
}

export function fitImageView(image, stage, limits = {}) {
  dimensions(image, stage, limits);
  const scale = Math.min(stage.width / image.width, stage.height / image.height) * FIT_MARGIN;
  const w = stage.width / scale;
  const h = stage.height / scale;
  return constrainView({x: (image.width - w) / 2, y: (image.height - h) / 2, w, h}, image, stage, limits);
}

export function actualImageView(image, stage, limits = {}) {
  dimensions(image, stage, limits);
  return constrainView({x: (image.width - stage.width) / 2, y: (image.height - stage.height) / 2, w: stage.width, h: stage.height}, image, stage, limits);
}
